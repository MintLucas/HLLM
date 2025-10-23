import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SequentialSampler
from transformers import AutoModel, AutoTokenizer
import math
import json
import os
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 改进点：ItemLLM的输入为tokenizer处理后的tensor，DataSet返回tokenizer的处理结果。



class ItemLLM(nn.Module):
    def __init__(self, model_name="bert-base-chinese",device = None):
        super(ItemLLM, self).__init__()
        self.device = device
        # 加载预训练语言模型
        hub_path = '/Users/gaolin3/workspace/supertopic_demo/hllm/hub'
        self.llm = AutoModel.from_pretrained(model_name, cache_dir=hub_path).to(self.device)
        


        self.item_token_id = 21128
        self.tokenizer_size = 21129
        self.llm.resize_token_embeddings(self.tokenizer_size)
        
        
    def forward(self, input_ids, attention_mask=None):
        """
        输入: 预处理后的token tensor (batch_size, seq_len)
        输出: 物品特征嵌入 (batch_size, hidden_size)
        """
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 构建模型输入
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
            
        # 获取模型输出
        outputs = self.llm(** inputs)
        
        # 提取特殊token对应的隐藏状态作为物品嵌入
        item_positions = (input_ids == self.item_token_id).nonzero()[:, 1]
        embeddings = outputs.last_hidden_state[torch.arange(len(item_positions), device=self.device), item_positions]

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
    def __init__(self, item_llm, user_llm, device=None):
        super(HLLM, self).__init__()
        self.item_llm = item_llm
        self.user_llm = user_llm
        self.device = device
        
    def forward(self, user_history_inputs, next_inputs=None, target_inputs=None):
        """
        输入: 
        - user_history_inputs: 历史物品的tokenized输入 dict containing input_ids and attention_mask
        - next_inputs/target_inputs: 下一个物品/目标物品的tokenized输入
        输出:
        - 生成式任务: 预测的下一个物品特征和真实特征
        - 判别式任务: 用户对目标物品的兴趣分数
        """
        # 处理历史物品特征
        hist_input_ids = user_history_inputs['input_ids']  # (batch_size, seq_len, max_seq_len)
        hist_attention_mask = user_history_inputs['attention_mask']  # (batch_size, seq_len, max_seq_len)
        batch_size, seq_len, max_seq_len = hist_input_ids.shape
        
        # 展平历史物品输入以批量处理
        flat_input_ids = hist_input_ids.view(-1, max_seq_len)  # (batch_size*seq_len, max_seq_len)
        flat_attention_mask = hist_attention_mask.view(-1, max_seq_len)  # (batch_size*seq_len, max_seq_len)
        
        # 获取历史物品嵌入
        print(flat_input_ids.shape)
        item_embs = self.item_llm(flat_input_ids, flat_attention_mask)  # (batch_size*seq_len, hidden_size)
        print(item_embs.shape)
        item_embs = item_embs.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_size)
        
        if target_inputs is None:
            # 生成式任务
            next_input_ids = next_inputs['input_ids']
            next_attention_mask = next_inputs['attention_mask']
            next_embs = self.item_llm(next_input_ids, next_attention_mask)
            predicted_embs = self.user_llm(item_embs)
            return predicted_embs, next_embs
        else:
            # 判别式任务
            target_input_ids = target_inputs['input_ids']
            target_attention_mask = target_inputs['attention_mask']
            target_emb = self.item_llm(target_input_ids, target_attention_mask)
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
# 修改后的RecommendationDataset类
class RecommendationDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length=20, max_seq_len=256):
        self.data = data
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.item_token = "[ITEM]"
        
    def __len__(self):
        return len(self.data)
    
    def _process_text(self, text):
        """处理单个文本并转换为tensor"""
        processed_text = f"Compress the following into embedding: {text[:200]} {self.item_token}"
        return self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len
        )
    
    def __getitem__(self, idx):
        user_data = self.data[idx]
        history = user_data['history'][-self.seq_length:]
        # 确保历史长度一致
        if len(history) < self.seq_length:
            history = [""] * (self.seq_length - len(history)) + history
            
        next_item = user_data['history'][-1] if user_data['history'] else ""
        
        # 处理历史文本
        hist_inputs = [self._process_text(text) for text in history]
        hist_input_ids = torch.cat([x['input_ids'] for x in hist_inputs], dim=0)
        hist_attention_mask = torch.cat([x['attention_mask'] for x in hist_inputs], dim=0)
        
        # 处理下一个物品
        next_inputs = self._process_text(next_item)
        
        return {
            'history_inputs': {
                'input_ids': hist_input_ids,
                'attention_mask': hist_attention_mask
            },
            'next_inputs': {
                'input_ids': next_inputs['input_ids'].squeeze(0),
                'attention_mask': next_inputs['attention_mask'].squeeze(0)
            },
            'target_inputs': {
                'input_ids': next_inputs['input_ids'].squeeze(0),
                'attention_mask': next_inputs['attention_mask'].squeeze(0)
            },
            'labels': torch.tensor(1.0, dtype=torch.float32)
        }

def train_discriminative(ddp_model, dataloader, sampler, optimizer, local_rank, seq_len, batch_size):
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(3):
        # sampler.set_epoch(epoch)
        # ddp_model.train()
        total_loss = 0
    
        for batch in dataloader:

            print( batch['history_inputs']['input_ids'].shape)
            print( batch['next_inputs']['input_ids'].shape)
            history_inputs = {k: v.to(ddp_model.device) for k, v in batch['history_inputs'].items()}
            next_inputs = {k: v.to(ddp_model.device) for k, v in batch['next_inputs'].items()}
            target_inputs = {k: v.to(ddp_model.device) for k, v in batch['target_inputs'].items()}
            labels = batch['labels'].to(ddp_model.device)

            # if len(history_texts) != seq_len or len(history_texts[0]) != batch_size or len(next_texts) != batch_size:
            #     continue
            # 前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_embs,next_embs = ddp_model(history_inputs, next_inputs=next_inputs)
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

            torch.save(checkpoint, "/Users/gaolin3/workspace/supertopic_demo/hllm/checkpoint."+str(epoch))


def main():


    batch_size = 8
    seq_len = 20
    hidden_size = 768
    local_rank=0
    # local_rank = int(os.environ["LOCAL_RANK"])
    # dist.init_process_group(backend="nccl")
    # torch.cuda.set_device(local_rank)
    device = torch.device("cpu")

    hub_path = '/Users/gaolin3/workspace/supertopic_demo/hllm/hub'

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese',cache_dir=hub_path)
    item_token = "[ITEM]"

    tokenizer.add_tokens([item_token])

    item_llm = ItemLLM(device=device)
    user_llm = UserLLM(hidden_size=hidden_size, device=device)
    hllm = HLLM(item_llm, user_llm).to(device)
    # ddp_model = DDP(hllm, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.SGD(hllm.parameters(), lr=0.001)

    data_path = '/Users/gaolin3/workspace/supertopic_demo/hllm/data.json'
    with open(data_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)  # 反序列化为 Python 字典


    dataset = RecommendationDataset(sample_data, tokenizer)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=True,
        sampler=sampler,
        shuffle=False)

    train_discriminative(hllm, dataloader,sampler, optimizer, local_rank, seq_len, batch_size)

if __name__ == "__main__":
    import os
    main()
