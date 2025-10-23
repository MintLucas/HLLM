import torch
import numpy as np
import re
import json
from transformers import AutoTokenizer, AutoModel
import triton_python_backend_utils as pb_utils

# 保持原有的文本清理工具函数
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

# 定义ItemLLM类（保持核心功能）
class ItemLLM(torch.nn.Module):
    def __init__(self, model_name="bert-base-chinese", device=None):
        super(ItemLLM, self).__init__()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 加载预训练语言模型
        #hub_path = 'hdfs://fed/user/wb_oprd_supertopic_algo/model/zoo/model_id=d_m_21802/v20250924/hub'
        hub_path = '/workspace/data/customize-bert/hub/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea'
        self.llm = AutoModel.from_pretrained(
            hub_path
            #model_name, 
            #cache_dir=hub_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_path
            #model_name,
            #cache_dir=hub_path
        )
        
        # 添加特殊token用于提取物品特征
        self.item_token = "[ITEM]"
        if self.item_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.item_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.item_token_id = self.tokenizer.convert_tokens_to_ids(self.item_token)
        
    def forward(self, text_descriptions):
        """生成文本对应的向量"""
        processed_texts = [
            f"Compress the following into embedding: {desc[:500]} {self.item_token}"
            for desc in text_descriptions
        ]
        
        inputs = self.tokenizer(
            processed_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm(** inputs)
        
        item_positions = (inputs.input_ids == self.item_token_id).nonzero()[:, 1]
        embeddings = outputs.last_hidden_state[
            torch.arange(len(item_positions), device=self.device), 
            item_positions
        ]
        return embeddings

# Triton模型类
class TritonPythonModel:
    def initialize(self, args):
        """模型初始化，加载模型权重"""
        self.model_config = model_config = json.loads(args["model_config"])
        
        # 获取输出配置
        output_config = pb_utils.get_output_config_by_name(
            model_config, "vector_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )
        
        # 加载模型
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint_path = "/workspace/data/customize-bert/checkpoint.2"  # 替换为实际路径
        self.item_llm = self._load_model(checkpoint_path)
        self.item_llm.eval()

    def _load_model(self, checkpoint_path):
        """加载模型 checkpoint"""
        item_llm = ItemLLM(device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 提取ItemLLM相关参数
        item_llm_state_dict = {
            k.replace('item_llm.', ''): v 
            for k, v in checkpoint['model_state_dict'].items() 
            if k.startswith('item_llm.')
        }
        
        item_llm.load_state_dict(item_llm_state_dict, strict=False)
        return item_llm

    def execute(self, requests):
        """处理推理请求"""
        responses = []
        
        for request in requests:
            # 获取输入文本
            input_texts = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
            input_texts = [text.decode('utf-8') for text in input_texts]
            
            # 文本预处理
            cleaned_texts = [clean_text(text) for text in input_texts]
            
            # 生成向量
            with torch.no_grad():
                vectors = self.item_llm(cleaned_texts)
            
            # 转换为numpy数组
            vectors_np = vectors.cpu().numpy().astype(self.output_dtype)
            
            # 创建输出张量
            output_tensor = pb_utils.Tensor("vector_output", vectors_np)
            
            # 构建响应
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """模型清理"""
        pass
