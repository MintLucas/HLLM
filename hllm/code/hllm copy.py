import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import math
import json
import os
import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ItemLLM(nn.Module):
    def __init__(self, model_name="bert-base-chinese",device = None):
        super(ItemLLM, self).__init__()
        self.device = device
        # 加载预训练语言模型
        self.llm = AutoModel.from_pretrained(model_name, cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub')
        
        # 添加特殊token用于提取物品特征
        self.item_token = "[ITEM]"
        self.tokenizer.add_tokens([self.item_token])
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.item_token_id = self.tokenizer.convert_tokens_to_ids(self.item_token)
        
    def forward(self, text_descriptions):
        """
        输入: 物品文本描述列表
        输出: 物品特征嵌入 (batch_size, hidden_size)
        """
        # 预处理文本：添加提示词和特殊token
        processed_texts = [
            f"Compress the following into embedding: {desc[:500]} {self.item_token}"
            for desc in text_descriptions
        ]
        
        # Tokenize并获取特殊token对应的隐藏状态
        inputs = self.tokenizer(processed_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.llm(**inputs)
        # 提取特殊token对应的隐藏状态作为物品嵌入
        item_positions = (inputs.input_ids == self.item_token_id).nonzero()[:, 1]
        embeddings = outputs.last_hidden_state[torch.arange(len(item_positions),device=self.device), item_positions]

        # if embeddings.shape[0] != 160:
        #     print(processed_texts)
        #     print(inputs['input_ids'].shape)
        #     print(item_positions,item_positions.shape)
        #     print(embeddings, embeddings.shape)
        return embeddings



class UserLLM(nn.Module):
    def __init__(self, hidden_size, num_layers=3, nhead=4, device = None):
        super(UserLLM, self).__init__()
        self.device = device

        self.hidden_size = hidden_size
        # 基于Transformer的序列建模
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=4*hidden_size,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, item_embeddings):
        """
        输入: 物品特征序列 (batch_size, seq_len, hidden_size)
        输出: 下一个物品的预测特征 (batch_size, hidden_size)
        """
        # 添加位置编码
        seq_len = item_embeddings.size(1)
        positions = torch.arange(seq_len, device=item_embeddings.device).float()
        position_emb = self.position_encoding(positions)  # 实现略
        
        # Transformer编码
        transformer_input = item_embeddings + position_emb
        sequence_output = self.transformer(transformer_input)
        
        # 预测下一个物品特征 (取最后一个位置的输出)
        last_output = sequence_output[:, -1, :]
        predicted_embedding = self.predictor(last_output)
        
        return predicted_embedding
    
    def position_encoding(self, x):

        if x.dim() != 1:
            raise ValueError(f"输入必须是1维tensor，当前维度: {x.dim()}")
    
        seq_len = x.size(0)
        device = x.device  # 保持与输入相同的设备
    
        # 初始化位置编码矩阵
        pe = torch.zeros(seq_len, self.hidden_size, device=device)
    
        # 计算位置索引与频率因子
        position = x.unsqueeze(1)  # 形状: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=device) * 
                         (-math.log(10000.0) / self.hidden_size))  # 形状: (d_model//2,)
    
        # 偶数维度使用正弦编码，奇数维度使用余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.hidden_size % 2 == 1:
            # 处理奇数维度情况，避免余弦部分越界
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
    
        return pe


class HLLM(nn.Module):
    def __init__(self, item_llm, user_llm, device = None):
        super(HLLM, self).__init__()
        self.item_llm = item_llm
        self.user_llm = user_llm
        
    def forward(self, user_history_texts, next_text=None,target_text=None):
        """
        输入: 
        - user_history_texts: 用户历史交互物品的文本描述列表 (batch_size, seq_len)
        - target_text: 目标物品文本描述 (batch_size,), 判别式任务需要
        输出:
        - 生成式任务: 预测的下一个物品特征
        - 判别式任务: 用户对目标物品的兴趣分数
        """
        # 通过Item LLM获取历史物品特征
        flat_texts = [item for tpl in user_history_texts for item in tpl]

        batch_size = len(user_history_texts[0])
        seq_len = len(user_history_texts)
        # [159, 768]
        item_embs = self.item_llm(flat_texts)
        bs, hs = item_embs.shape
        print('3', bs, hs, batch_size, seq_len)
        item_embs = item_embs.reshape(batch_size, seq_len, -1)
        

        if target_text is None:
            # 生成式任务: 预测下一个物品
            return self.user_llm(item_embs), self.item_llm(next_text)
        else:
            # 判别式任务: 计算用户特征与目标物品特征的相似度
            target_emb = self.item_llm(target_text)
            user_emb = self.user_llm(item_embs)
            return torch.cosine_similarity(user_emb, target_emb, dim=-1)






def info_nce_loss(predicted, positive, temperature=0.1, num_negatives=10):
    """
    InfoNCE对比损失函数
    predicted: (batch_size, hidden_size)
    positive: (batch_size, hidden_size)
    """
    # 随机采样负样本 (简化实现)
    batch_size = predicted.size(0)
    neg_indices = torch.randint(0, batch_size, (batch_size, num_negatives))
    negatives = positive[neg_indices]  # (batch_size, num_negatives, hidden_size)
    
    # 计算相似度
    pos_sim = torch.cosine_similarity(predicted, positive, dim=-1) / temperature
    neg_sim = torch.cosine_similarity(
        predicted.unsqueeze(1),
        negatives,
        dim=-1
    ) / temperature
    
    # 计算对比损失
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=predicted.device)
    return nn.CrossEntropyLoss()(logits, labels)


import re

def remove_emoji(content):
    if len(content) > 0:
        content = re.sub('(\\[[^\\]]*\\])','', content)

        emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号 & 标志
        "\U0001F680-\U0001F6FF"  # 运输 & 地图符号
        "\U0001F1E0-\U0001F1FF"  # 国旗 (iOS)
        "\U00002700-\U000027BF"  # 杂项符号和箭头
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U00002600-\U000026FF"  # 杂项符号
        "\U00002B50-\U00002BFF"  # 符号和箭头
        "]+", flags=re.UNICODE)

        content = emoji_pattern.sub(r'', content)
    return content


def clean_text(text):
    """
    全面清理字符串中的各类杂质
    
    功能包括：
    - 去除网址
    - 去除emoji表情包
    - 清理\u200b、\n等特殊符号
    - 清理连续过多空格
    - 清理戳>>、详情>>等特殊文字
    """
    # 1. 去除网址 (http/https/www开头的链接)
    url_pattern = re.compile(r'https?://\S+|www\.\S+|https?://|www\.')
    text = url_pattern.sub('', text)
    
    text = remove_emoji(text)
    
    # 3. 清理特殊符号 (\u200b是零宽空格，\n是换行等)
    special_chars = re.compile(r'[\u200b\n\r\t]+')
    text = special_chars.sub(' ', text)
    
    # 4. 清理指定特殊文字 (戳>>、详情>>等)
    special_texts = re.compile(r'(戳>>|详情>>|点击>>|查看>>|链接>>)')
    text = special_texts.sub('', text)
    
    # 5. 清理连续过多空格 (合并为单个空格并去除首尾空格)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

from torch.utils.data import Dataset, DataLoader

class RecommendationDataset(Dataset):
    def __init__(self, data, seq_length=20):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_data = self.data[idx]
        
        # 生成训练序列
        history = user_data['history'][-self.seq_length:]
        history = [clean_text(s)[:100] for s in history]
        next_item = user_data['history'][-1] if len(user_data['history']) > self.seq_length else user_data['history'][-1]
        
        return {
            'history_texts': history,
            'next_texts': next_item,
            'target_texts': next_item,  # 用于判别式任务
            'labels': 1.0  # 正样本
        }

def train_generative(ddp_model, dataloader, sampler, optimizer, local_rank):
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0
    
        for batch in dataloader:
            # 获取批次数据
            history_texts = batch['history_texts']  # (batch_size, seq_len)
            next_texts = batch['next_texts']  # (batch_size,)
            if len(history_texts) != seq_len or len(history_texts[0]) != batch_size:
                continue
            # 前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_embs,next_embs = ddp_model(history_texts, next_text=next_texts)
                # 计算InfoNCE损失
                loss = info_nce_loss(predicted_embs, next_embs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # 反向传播
            #loss.backward()
            #optimizer.step()
        
            total_loss += loss.item()
            print('batch loss: ' + str(loss.item()))

        if local_rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")


def train_discriminative(ddp_model, dataloader, sampler, optimizer, local_rank, seq_len, batch_size):
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0
    
        for batch in dataloader:
            # 获取批次数据
            history_texts = batch['history_texts']  # (batch_size, seq_len)
            target_texts = batch['target_texts']  # (batch_size,)
            next_texts = batch['next_texts']  # (batch_size,)
            labels = batch['labels']  # (batch_size,)
            if len(history_texts) != seq_len or len(history_texts[0]) != batch_size or len(next_texts) != batch_size:
                continue
            # 前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_embs,next_embs = ddp_model(history_texts, next_text=next_texts)
                # 计算InfoNCE损失
                loss = info_nce_loss(predicted_embs, next_embs)

                #logits = ddp_model(history_texts, target_texts)
                # 计算交叉熵损失
                #bce_loss = nn.BCEWithLogitsLoss()(logits, labels.float())

                #loss = bce_loss + 0.5 * info_nce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # 反向传播
            #loss.backward()
            #optimizer.step()
        
            total_loss += loss.item()
            print('batch loss: ' + str(loss.item()))

        if local_rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }

            torch.save(checkpoint, "/njfs/train-comment/example/gaolin3/hllm/checkpoint."+str(epoch))

def local_set():

    # --- 调试时手动设置环境变量 ---
    # 注意：这些设置通常只用于单进程调试或特殊情况。
    # 在生产环境中，推荐使用 torchrun。

    # 1. 进程排名 (RANK): 必须设置，这里设置为 0 (第一个进程)
    os.environ['RANK'] = '0' 
    os.environ["LOCAL_RANK"] = '0'
    # 2. 总进程数 (WORLD_SIZE): 必须设置，这里设置为 1 (只运行一个进程进行调试)
    os.environ['WORLD_SIZE'] = '1'
    
    # 3. 主节点地址 (MASTER_ADDR): 必须设置，可以使用本地回环地址
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    
    # 4. 主节点端口 (MASTER_PORT): 必须设置，选择一个未被占用的端口
    os.environ['MASTER_PORT'] = '29500' # 29500 是 PyTorch 分布式常用的默认端口
    
    print(f"DEBUG: Setting RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")
    # ------------------------------------

    try:
        # 你的原始代码:
        # File "/data2/zhipeng16/git/HLLM/hllm/code/hllm.py", line 365, in main
        dist.init_process_group(backend="nccl") 
        print("Distributed group initialized successfully.")
        
        # ... 后续的训练/推理代码 ...

    except ValueError as e:
        print(f"Error during init_process_group: {e}")
        # 如果你设置了，但仍然报错，可以检查其他环境变量或配置
        
    # ...
    pass # 示例中省略了 main 函数的其余部分

def main():

    local_set()
    batch_size = 8
    seq_len = 20
    hidden_size = 768
    # local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    item_llm = ItemLLM(device=device)
    user_llm = UserLLM(hidden_size=hidden_size, device=device)
    hllm = HLLM(item_llm, user_llm).to(device)
    ddp_model = DDP(hllm, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    with open("/njfs/train-comment/example/gaolin3/hllm/data.json", "r", encoding="utf-8") as f:
        sample_data = json.load(f)  # 反序列化为 Python 字典


    dataset = RecommendationDataset(sample_data)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=True,
        sampler=sampler,
        shuffle=False)



    # scaler = torch.cuda.amp.GradScaler()

    # for epoch in range(3):
    #     sampler.set_epoch(epoch)
    #     ddp_model.train()
    #     total_loss = 0
    
    #     for batch in dataloader:
    #         # 获取批次数据
    #         history_texts = batch['history_texts']  # (batch_size, seq_len)
    #         next_texts = batch['next_texts']  # (batch_size,)
    #         if len(history_texts) != seq_len or len(history_texts[0]) != batch_size:
    #             continue
    #         # 前向传播
    #         optimizer.zero_grad()
    #         with torch.cuda.amp.autocast():
    #             predicted_embs,next_embs = ddp_model(history_texts, next_text=next_texts)
    #             # 计算InfoNCE损失
    #             loss = info_nce_loss(predicted_embs, next_embs)

    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         # 反向传播
    #         #loss.backward()
    #         #optimizer.step()
        
    #         total_loss += loss.item()
    #         print('batch loss: ' + str(loss.item()))

    #     if local_rank == 0:
    #         print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

    train_discriminative(ddp_model, dataloader,sampler, optimizer, local_rank, seq_len, batch_size)
    dist.destroy_process_group()

if __name__ == "__main__":
    import os
    main()
