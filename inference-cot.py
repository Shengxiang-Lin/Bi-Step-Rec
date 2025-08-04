import os
import json
import torch
import logging
import re
import time
from tqdm import tqdm
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class BatchKeywordGenerator:
    """批量游戏关键词生成器"""
    def __init__(self, model_name: str = "base_models/Qwen2-7B-Instruct", batch_size: int = 16, max_items: int = None):
        """
        初始化关键词生成器
        参数:
            model_name: 使用的LLM模型名称
            batch_size: 批量处理大小
            max_items: 最大处理条目数 (None表示处理全部)
        """
        self.batch_size = batch_size
        self.max_items = max_items
        # 加载模型
        self.load_keyword_model(model_name)
        logger.info(f"Game Keyword Generator initialized with batch size {batch_size}, max items {max_items if max_items is not None else 'all'}")
    
    def load_keyword_model(self, model_name: str):
        """加载关键词生成模型"""
        try:
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置pad token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info(f"Loaded keyword generation model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_keyword_prompt(self, game_name: str) -> str:
        """获取关键词提示模板"""
        return f"""<|im_start|>system
You are a professional text analysis expert. Your task is to generate a detailed 
chain of thought analysis based on the item's textual description according to the 
five requirements listed below, and then summarize keywords that describe the 
item's characteristics. Each keyword must focus on the item's attributes.
Please analyze objectively and rationally.
1. Identify the first key characteristic of the item and cite the exact text that supports this characteristic.
2. Identify the second key characteristic of the item and cite the exact text that supports this characteristic.
3. Identify the third key characteristic of the item and cite the exact text that supports this characteristic.
4. Identify the fourth key characteristic of the item and cite the exact text that supports this characteristic.
5. Identify the fifth key characteristic of the item and cite the exact text that supports this characteristic.
Provide response in strict JSON format:{{"CoT":"Step-by-step analysis with source 
text references","keywords":"Comma-separated list of keywords"}}<|im_end|>
<|im_start|>user
The text description of an item is as follows:
{game_name}<|im_end|>
<|im_start|>assistant
"""
    
    def batch_generate_keywords(self, game_names: List[str]) -> List[str]:
        """批量生成关键词"""
        if not game_names:
            return []
        
        try:
            # 为所有游戏名称生成提示
            prompts = [self.get_keyword_prompt(name) for name in game_names]
            
            # 批量编码输入
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,  # 启用填充
                truncation=True,
                max_length=512,
                pad_to_multiple_of=8  # 优化GPU性能
            ).to(self.model.device)
            
            # 生成配置
            generation_config = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.1,
                "top_p": 0.9,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 批量解码输出
            responses = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=False
            )
            
            # 处理每个响应
            keywords_list = []
            for response in responses:
                # 找到assistant部分的开始
                assistant_start = response.find("<|im_start|>assistant")
                if assistant_start != -1:
                    assistant_response = response[assistant_start:]
                    # 提取JSON对象
                    json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                            # 成功提取关键词
                            keywords_list.append(result.get("keywords", ""))
                            continue
                        except json.JSONDecodeError:
                            # 尝试直接提取keywords字段
                            keywords_match = re.search(r'"keywords":\s*"([^"]+)"', assistant_response)
                            if keywords_match:
                                keywords_list.append(keywords_match.group(1))
                                continue
                
                # 如果提取失败，返回原始游戏名称
                # 从响应中提取原始游戏名称
                game_name_match = re.search(r'The text description of an item is as follows:\n([^\n]+)', response)
                if game_name_match:
                    game_name = game_name_match.group(1).strip()
                    keywords_list.append(game_name)
                else:
                    # 如果无法提取游戏名称，使用最后一个游戏名称作为备选
                    keywords_list.append(game_names[-1] if game_names else "")
            
            return keywords_list
        
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            # 返回原始游戏名称列表作为备选
            return game_names
    
    def process_json_file(self, input_path: str, output_path: str):
        """处理JSON文件并添加关键词"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 读取输入文件
            with open(input_path, 'r') as f:
                full_data = json.load(f)
            
            logger.info(f"Loaded {len(full_data)} records from {input_path}")
            
            # 确定要处理的数据范围
            if self.max_items is not None:
                data = full_data[:self.max_items]
                logger.info(f"Processing first {len(data)} items")
            else:
                data = full_data
                logger.info(f"Processing all {len(data)} items")
            
            # 准备所有游戏名称
            all_game_names = []
            for item in data:
                # 提取游戏名称 (从output字段)
                game_name = item["predict"][0]
                all_game_names.append(game_name)
            
            # 批量处理所有游戏名称
            total_batches = (len(all_game_names) + self.batch_size - 1) // self.batch_size
            all_keywords = []
            
            # 进度条
            pbar = tqdm(total=len(all_game_names), desc="Generating keywords")
            
            # 分批处理
            for i in range(0, len(all_game_names), self.batch_size):
                batch_names = all_game_names[i:i + self.batch_size]
                
                # 记录时间
                start_time = time.time()
                
                # 生成关键词
                batch_keywords = self.batch_generate_keywords(batch_names)
                
                # 计算处理速度
                batch_time = time.time() - start_time
                speed = len(batch_names) / batch_time if batch_time > 0 else 0
                
                # 添加到结果
                all_keywords.extend(batch_keywords)
                
                # 更新进度条
                pbar.update(len(batch_names))
                
                # 日志记录
                logger.info(f"Processed batch {i//self.batch_size+1}/{total_batches}: "
                            f"{len(batch_names)} items at {speed:.2f} items/sec")
            
            pbar.close()
            
            # 将关键词分配到每个条目
            for item, keywords in zip(data, all_keywords):
                item["keywords"] = keywords
            
            # 保存结果
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved results to {output_path} with {len(data)} items")
            return True
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return False

if __name__ == "__main__":
    # 初始化生成器
    batch_size = 128
    
    # 设置处理数量 (None表示处理全部)
    max_items = None  # 只处理前100条
    
    generator = BatchKeywordGenerator(
        model_name="base_models/Qwen2-7B-Instruct",
        batch_size=batch_size,
        max_items=max_items
    )
    
    # 定义文件路径
    input_file = "data/game/result/base/llama-7b-7B-lora-alpaca-game-base-0-40.json"
    output_file = "data/game/result/base/llama-7b-7B-lora-alpaca-game-base-0-40-keywords.json"
    
    # 处理文件
    generator.process_json_file(input_file, output_file)