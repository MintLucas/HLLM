import torch
from transformers import AutoTokenizer, AutoModel
import re

# 定义必要的工具函数（与原代码保持一致）
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
    """清理文本，与训练时保持一致的预处理"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+|https?://|www\.')
    text = url_pattern.sub('', text)
    text = remove_emoji(text)
    special_chars = re.compile(r'[\u200b\n\r\t]+')
    text = special_chars.sub(' ', text)
    special_texts = re.compile(r'(戳>>|详情>>|点击>>|查看>>|链接>>)')
    text = special_texts.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 定义ItemLLM类（与原代码保持一致）
class ItemLLM(torch.nn.Module):
    def __init__(self, model_name="bert-base-chinese", device=None):
        super(ItemLLM, self).__init__()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 加载预训练语言模型
        self.llm = AutoModel.from_pretrained(
            model_name, 
            cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub'
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir='/njfs/train-comment/example/gaolin3/hllm/hub'
        )
        
        # 添加特殊token用于提取物品特征
        self.item_token = "[ITEM]"
        # 检查是否已包含该token，避免重复添加
        if self.item_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.item_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.item_token_id = self.tokenizer.convert_tokens_to_ids(self.item_token)
        
    def forward(self, text_descriptions):
        """生成文本对应的向量"""
        # 预处理文本：添加提示词和特殊token
        processed_texts = [
            f"Compress the following into embedding: {desc[:500]} {self.item_token}"
            for desc in text_descriptions
        ]
        
        # Tokenize并获取特殊token对应的隐藏状态
        inputs = self.tokenizer(
            processed_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():  # 推理时不计算梯度
            outputs = self.llm(**inputs)
        
        # 提取特殊token对应的隐藏状态作为物品嵌入
        item_positions = (inputs.input_ids == self.item_token_id).nonzero()[:, 1]
        embeddings = outputs.last_hidden_state[
            torch.arange(len(item_positions), device=self.device), 
            item_positions
        ]
        return embeddings

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

def text_to_vector(text, item_llm):
    """将单段文本转换为向量"""
    # 文本预处理
    cleaned_text = clean_text(text)
    # 转换为向量（注意输入需要是列表形式）
    with torch.no_grad():
        vector = item_llm([cleaned_text])
    # 转换为numpy数组并返回
    return vector.cpu().numpy()[0]  # 取第一个元素（因为输入是单条文本）

# 使用示例
if __name__ == "__main__":
    # 配置
    checkpoint_path = "/njfs/train-comment/example/gaolin3/hllm/checkpoint.2"  # 替换为实际的checkpoint路径
    test_text = "这是一个测试商品，价格实惠，质量很好，推荐购买！"  # 测试文本
    
    # 加载模型
    print("加载ItemLLM模型...")
    item_llm = load_item_llm_from_checkpoint(checkpoint_path)
    
    # 生成向量
    print("生成文本向量...")
    vector = text_to_vector(test_text, item_llm)
    
    # 输出结果信息
    print(f"文本向量形状: {vector.shape}")
    print(f"向量前5个元素: {vector[:5]}")
