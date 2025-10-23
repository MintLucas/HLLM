import torch
import re
from transformers import AutoTokenizer, AutoModel

# 复用原文件中的文本清理函数
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
    url_pattern = re.compile(r'https?://\S+|www\.\S+|https?://|www\.')
    text = url_pattern.sub('', text)
    text = remove_emoji(text)
    special_chars = re.compile(r'[\u200b\n\r\t]+')
    text = special_chars.sub(' ', text)
    special_texts = re.compile(r'(戳>>|详情>>|点击>>|查看>>|链接>>)')
    text = special_texts.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 复用原文件中的模型类定义
class ItemLLM(torch.nn.Module):
    def __init__(self, model_name="bert-base-chinese", device=None):
        super(ItemLLM, self).__init__()
        self.device = device
        self.llm = AutoModel.from_pretrained(model_name, cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub')
        self.item_token = "[ITEM]"
        self.tokenizer.add_tokens([self.item_token])
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.item_token_id = self.tokenizer.convert_tokens_to_ids(self.item_token)
        
    def forward(self, text_descriptions):
        processed_texts = [f"Compress the following into embedding: {desc[:500]} {self.item_token}" for desc in text_descriptions]
        inputs = self.tokenizer(processed_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.llm(** inputs)
        item_positions = (inputs.input_ids == self.item_token_id).nonzero()[:, 1]
        embeddings = outputs.last_hidden_state[torch.arange(len(item_positions), device=self.device), item_positions]
        return embeddings

class UserLLM(torch.nn.Module):
    def __init__(self, hidden_size, num_layers=3, nhead=4, device=None):
        super(UserLLM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=4*hidden_size,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, item_embeddings):
        seq_len = item_embeddings.size(1)
        positions = torch.arange(seq_len, device=item_embeddings.device).float()
        position_emb = self.position_encoding(positions)
        transformer_input = item_embeddings + position_emb
        sequence_output = self.transformer(transformer_input)
        last_output = sequence_output[:, -1, :]
        predicted_embedding = self.predictor(last_output)
        return predicted_embedding
    
    def position_encoding(self, x):
        if x.dim() != 1:
            raise ValueError(f"输入必须是1维tensor，当前维度: {x.dim()}")
        seq_len = x.size(0)
        device = x.device
        pe = torch.zeros(seq_len, self.hidden_size, device=device)
        position = x.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=device) * (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.hidden_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class HLLM(torch.nn.Module):
    def __init__(self, item_llm, user_llm, device=None):
        super(HLLM, self).__init__()
        self.item_llm = item_llm
        self.user_llm = user_llm
        
    def forward(self, user_history_texts, next_text=None, target_text=None):
        flat_texts = [item for tpl in user_history_texts for item in tpl]
        batch_size = len(user_history_texts[0])
        seq_len = len(user_history_texts)
        item_embs = self.item_llm(flat_texts)
        bs, hs = item_embs.shape
        item_embs = item_embs.reshape(batch_size, seq_len, -1)
        if target_text is None:
            return self.user_llm(item_embs), self.item_llm(next_text)
        else:
            target_emb = self.item_llm(target_text)
            user_emb = self.user_llm(item_embs)
 
            return torch.cosine_similarity(user_emb, target_emb, dim=-1)


def load_item_llm_from_checkpoint(checkpoint_path, model_name="bert-base-chinese", device=None):
    """从训练好的模型 checkpoint 加载 ItemLLM"""
    device = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 初始化ItemLLM
    item_llm = ItemLLM(model_name=model_name, device=device)
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取ItemLLM相关的参数（原模型是HLLM，包含item_llm子模块）
    item_llm_state_dict = {
        k.replace('item_llm.', ''): v 
        for k, v in checkpoint['model_state_dict'].items() 
        if k.startswith('item_llm.')
    }
    
    # 加载参数到ItemLLM
    item_llm.load_state_dict(item_llm_state_dict, strict=True)
    item_llm.eval()  # 设置为评估模式
    return item_llm


# 加载模型并使用UserLLM的核心函数
def load_user_llm_from_checkpoint(checkpoint_path, hidden_size=768, num_layers=3, nhead=4, device=None):
    """从 checkpoint 加载 UserLLM 模型"""
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化UserLLM
    user_llm = UserLLM(hidden_size=hidden_size, num_layers=num_layers, nhead=nhead, device=device)
    user_llm = user_llm.to(device)

    item_llm = ItemLLM(model_name='bert-base-chinese', device=device)
    item_llm = item_llm.to(device)
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取UserLLM相关的参数（原模型是HLLM，包含user_llm子模块）
    user_llm_state_dict = {
        k.replace('user_llm.', ''): v 
        for k, v in checkpoint['model_state_dict'].items() 
        if k.startswith('user_llm.')
    }
    
    # 加载参数到UserLLM
    user_llm.load_state_dict(user_llm_state_dict, strict=True)
    user_llm.eval()  # 设置为评估模式

    # 提取ItemLLM相关的参数（原模型是HLLM，包含item_llm子模块）
    item_llm_state_dict = {
        k.replace('item_llm.', ''): v 
        for k, v in checkpoint['model_state_dict'].items() 
        if k.startswith('item_llm.')
    }
    
    # 加载参数到ItemLLM
    item_llm.load_state_dict(item_llm_state_dict, strict=True)
    item_llm.eval()  # 设置为评估模式
    return user_llm,item_llm

def text_list_to_user_vector(text_list, user_llm, item_llm):
    """
    将文本列表（用户历史序列）转换为UserLLM输出向量
    
    参数:
        text_list: 字符串列表，代表用户历史交互的物品文本描述
        user_llm: 加载好的UserLLM模型
        item_llm: 用于将文本转换为物品向量的ItemLLM模型
    返回:
        用户序列向量 (hidden_size,)
    """
    # 文本预处理
    cleaned_texts = [clean_text(text) for text in text_list]
    
    # 先通过ItemLLM将文本转换为物品向量
    with torch.no_grad():
        item_embeddings = item_llm(cleaned_texts)  # 形状: (seq_len, hidden_size)
    
    # 调整形状为 (batch_size=1, seq_len, hidden_size)，适应UserLLM输入
    item_embeddings = item_embeddings.to(user_llm.device)
    item_embeddings = item_embeddings.unsqueeze(0)  # 增加批次维度
    
    # 通过UserLLM生成用户序列向量
    with torch.no_grad():
        user_vector = user_llm(item_embeddings)  # 形状: (1, hidden_size)
    
    # 去除批次维度并转换为numpy数组
    return user_vector.squeeze(0).cpu().numpy()

# 使用示例
if __name__ == "__main__":
    import math  # UserLLM的position_encoding需要math模块
    
    # 配置
    checkpoint_path = "/njfs/train-comment/example/gaolin3/hllm/checkpoint.2"  # 替换为实际checkpoint路径
    test_text_list = [
        "这是第一个商品，质量很好",
        "第二个商品价格实惠，推荐购买",
        "第三个商品外观精美，使用方便"
    ]  # 用户历史文本序列
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载ItemLLM（用于将文本转换为物品向量）
    print("加载ItemLLM模型...")
    #item_llm = load_item_llm_from_checkpoint(checkpoint_path)
    
    # 加载UserLLM
    print("加载UserLLM模型...")
    user_llm,item_llm = load_user_llm_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hidden_size=768,  # 需与训练时保持一致
        num_layers=3,     # 需与训练时保持一致
        nhead=4,          # 需与训练时保持一致
        device=device
    )
    
    # 生成用户序列向量
    print("生成用户序列向量...")
    user_vector = text_list_to_user_vector(test_text_list, user_llm, item_llm)
    
    # 输出结果信息
    print(f"用户序列向量形状: {user_vector.shape}")
    print(f"向量前5个元素: {user_vector[:5]}")
