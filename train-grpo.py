import os
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
os.environ["WANDB_DISABLED"] = "true"
os.environ['NCCL_SOCKET_TIMEOUT'] = '7200'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" 
import warnings
warnings.filterwarnings("ignore")
import sys
import json
import fire
import random
import torch
import logging
import datetime
import numpy as np
import transformers
from typing import List, Dict, Any, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from sentence_transformers import SentenceTransformer
import re
# 添加本地 TRL 路径到系统路径
trl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trl"))
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)
# 现在导入 GRPO 组件
from trl import GRPOTrainer, GRPOConfig
import torch.nn.functional as F
import warnings

# 配置日志系统
def setup_logging(output_dir):
    # 创建logs目录
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # 生成基于时间的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"llama_grpo_training_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_filepath

class RewardCalculator:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 阈值参数
        self.theta_FN = 0.85  # 假负相似度阈值
        self.theta_easy = 0.6  # 简单负样本阈值
        self.theta_FP = 0.75  # 假正相似度阈值
        self.theta_easy_low = 0.3  # 简单负样本低阈值
        
        # 奖励权重
        self.w1 = 0.8   # 困难负样本奖励
        self.w2 = 1.5   # 假负样本惩罚
        self.w3 = 1.5   # 假正样本惩罚
        self.w4 = 0.8   # 简单负样本惩罚
        self.w5 = 0.3   # 自适应温度奖励权重
        
        # 多样性奖励权重
        self.diversity_weight = 0.6
        
        # 用于动态调整阈值
        self.reward_history = []
        self.adjustment_counter = 0
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个游戏名称之间的余弦相似度"""
        embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
        
        # 添加随机扰动 (10%概率)
        if random.random() < 0.1:
            perturbation = 0.2 * (random.random() - 0.5)  # ±10%扰动
            similarity = min(1.0, max(-1.0, similarity + perturbation))
        
        return (similarity + 1) / 2  # 归一化到[0, 1]
    
    def extract_quoted_games(self, text: str) -> List[str]:
        """提取所有被双引号包裹的游戏名称"""
        # 使用正则表达式匹配双引号内的内容
        return re.findall(r'"([^"]+)"', text)
    
    def calculate_reward(self, user_history: str, generated_output: str) -> float:
        """
        计算生成商品的奖励 - 仅基于双引号包裹的游戏名称
        """
        # 从用户历史中提取游戏名称
        history_games = self.extract_quoted_games(user_history)
        
        # 从生成的输出中提取游戏名称
        gen_games = self.extract_quoted_games(generated_output)
        gen_game = gen_games[0] if gen_games else generated_output.strip()
        
        # 计算与用户历史整体的相似度
        history_text = " ".join(history_games)  # 将历史游戏连接成一个字符串
        sim_to_history = self.compute_similarity(history_text, gen_game)
        
        # 计算与每个历史游戏的最大相似度
        sim_values = [self.compute_similarity(gen_game, game) for game in history_games]
        sim_to_positive = max(sim_values) if sim_values else 0.0
        
        # 1. 奖励困难负样本
        if self.theta_easy < sim_to_history < self.theta_FN and sim_to_positive < self.theta_FP:
            rhard = self.w1
            logging.debug(f"Hard negative reward: {self.w1}")
        else:
            rhard = 0.0
        
        # 2. 惩罚假负样本
        rfalse = 0.0
        if sim_to_history >= self.theta_FN:
            rfalse -= self.w2
            logging.debug(f"False negative penalty: -{self.w2}")
        if sim_to_positive >= self.theta_FP:
            rfalse -= self.w3
            logging.debug(f"False positive penalty: -{self.w3}")
        
        # 3. 惩罚简单负样本
        reasy = -self.w4 if sim_to_history <= self.theta_easy_low else 0.0
        if reasy < 0:
            logging.debug(f"Easy negative penalty: -{self.w4}")
        
        # 4. 自适应温度奖励
        diversity = len(set(history_games)) / max(1, len(history_games))
        T_ideal = 1 / (1 + np.log(1 + 1/max(diversity, 0.001)))
        rtau = -self.w5 * abs(0.7 - T_ideal)
        
        # 5. 多样性奖励
        rdiversity = 0.0
        if len(history_games) >= 3:
            avg_sim = np.mean(sim_values) if sim_values else 0
            if 0.5 <= avg_sim <= 0.7:
                rdiversity = self.diversity_weight * (1 - abs(avg_sim - 0.6))
            logging.debug(f"Diversity reward: {rdiversity:.2f}")
        
        # 总奖励
        total_reward = rhard + rfalse + reasy + rtau + rdiversity
        
        # 记录奖励历史用于动态调整
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        # 动态调整阈值
        self.adjustment_counter += 1
        if self.adjustment_counter >= 100:
            self.adjust_thresholds()
            self.adjustment_counter = 0
        
        # 日志记录
        logging.debug(f"Generated game: {gen_game}")
        logging.debug(f"User history games: {history_games}")
        logging.debug(f"Sim to history: {sim_to_history:.4f}, Max sim to positive: {sim_to_positive:.4f}")
        logging.debug(f"Rhard: {rhard:.2f}, Rfalse: {rfalse:.2f}, Reasy: {reasy:.2f}, Rtau: {rtau:.2f}, Rdiversity: {rdiversity:.2f}, Total: {total_reward:.2f}")
        
        return total_reward
    
    def adjust_thresholds(self):
        """根据最近100个样本的奖励表现动态调整阈值"""
        if len(self.reward_history) < 50:
            return
        
        avg_reward = np.mean(self.reward_history)
        logging.info(f"Adjusting thresholds based on average reward: {avg_reward:.2f}")
        
        # 如果平均奖励过高，收紧阈值
        if avg_reward > 0.85:
            self.theta_FN = min(0.95, self.theta_FN + 0.01)
            self.theta_FP = min(0.85, self.theta_FP + 0.01)
            self.theta_easy = min(0.7, self.theta_easy + 0.01)
            logging.info(f"Tightening thresholds: FN={self.theta_FN:.2f}, FP={self.theta_FP:.2f}, Easy={self.theta_easy:.2f}")
        
        # 如果平均奖励过低，放宽阈值
        elif avg_reward < 0.4:
            self.theta_FN = max(0.7, self.theta_FN - 0.01)
            self.theta_FP = max(0.6, self.theta_FP - 0.01)
            self.theta_easy = max(0.4, self.theta_easy - 0.01)
            logging.info(f"Loosening thresholds: FN={self.theta_FN:.2f}, FP={self.theta_FP:.2f}, Easy={self.theta_easy:.2f}")

# 使用新的提示模板格式
def generate_prompt(data_point: Dict[str, Any]) -> str:
    """生成完整提示模板（包含输入和输出）"""
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

def generate_prediction_prompt(data_point: Dict[str, Any]) -> str:
    """生成预测提示模板（只包含输入）"""
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:"""

class SavePeftModelCallback(TrainerCallback):
    """自定义回调函数用于保存PEFT模型"""
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        logging.info(f"Saved adapter model to: {peft_model_path}")

class EpochLoggingCallback(TrainerCallback):
    """在每个epoch结束时记录日志"""
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = datetime.datetime.now()
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting Epoch {epoch_num}")
        logging.info(f"{'='*50}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_end_time = datetime.datetime.now()
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        
        if self.epoch_start_time:
            epoch_duration = epoch_end_time - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            avg_epoch_time = sum(self.epoch_times, datetime.timedelta(0)) / len(self.epoch_times)
            
            # 计算剩余时间估计
            remaining_epochs = args.num_train_epochs - epoch_num
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            logging.info(f"\n{'='*50}")
            logging.info(f"Completed Epoch {epoch_num}")
            logging.info(f"Epoch Duration: {epoch_duration}")
            logging.info(f"Average Epoch Time: {avg_epoch_time}")
            logging.info(f"Estimated Remaining Time: {estimated_remaining}")
            logging.info(f"{'='*50}\n")
        
        # 重置计时器
        self.epoch_start_time = None

def train_grpo(
    base_model_path: str = "base_models/llama-7b",  # 基础模型路径
    sft_lora_weights: str = "./llama-7b-lora-alpaca-game-base-0/checkpoint-40",  # SFT阶段保存的LoRA权重
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    output_dir: str = "./llama-7b-grpo-recommender",
    seed: int = 42,
    batch_size: int = 32,
    num_epochs: int = 3,  # RL训练通常需要较少轮次
    learning_rate: float = 1e-6,
    cutoff_len: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_sample: int = 256,  # 训练集采样数量，-1表示使用全部
    val_sample: int = 1000 
):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志系统
    log_filepath = setup_logging(output_dir)
    print(f"Logging GRPO training process to: {log_filepath}")
    logging.info(f"GRPO training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using Llama model: {base_model_path}")
    logging.info(f"Training Parameters: Epochs={num_epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator()
    
    # 加载数据集
    def load_data(paths: List[str], is_train: bool = False, sample: int = -1) -> Dataset:
        datasets = []
        for path in paths:
            logging.info(f"Loading data from: {path}")
            if path.endswith(".json"):
                dataset = load_dataset("json", data_files=path)["train"]
            else:
                dataset = load_dataset(path)["train"]
            datasets.append(dataset)
        
        concatenated = concatenate_datasets(datasets)
        
        # 添加采样逻辑
        if is_train and sample > -1:
            logging.info(f"Sampling {sample} examples from training dataset")
            concatenated = concatenated.shuffle(seed=seed).select(range(sample))
        elif is_train:
            logging.info("Shuffling full training dataset")
            concatenated = concatenated.shuffle(seed=seed)
        elif sample > -1:  # 验证集采样
            logging.info(f"Sampling {sample} examples from validation dataset")
            concatenated = concatenated.shuffle(seed=seed).select(range(sample))
        
        logging.info(f"Final samples: {len(concatenated)}")
        return concatenated
    
    # 加载数据时传入采样参数
    train_data = load_data(train_data_path, is_train=True, sample=train_sample)
    val_data = load_data(val_data_path, sample=val_sample)
    # 定义奖励函数 - 只处理Llama格式
    def reward_fn(prompts: List[str], completions: List[str]) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                # 从提示中提取用户历史 - 只处理Alpaca格式
                input_start = prompt.find("### Input:") + len("### Input:")
                input_end = prompt.find("### Response:", input_start)
                user_history = prompt[input_start:input_end].strip()
                
                # 计算奖励
                reward = reward_calculator.calculate_reward(user_history, completion)
                rewards.append(reward)
            except Exception as e:
                logging.error(f"Error calculating reward: {e}")
                rewards.append(0.0)  # 默认奖励
        
        # 记录批次奖励统计
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            avg_reward = sum(rewards) / len(rewards)
            logging.info(f"Batch rewards - Min: {min_reward:.2f}, Max: {max_reward:.2f}, Avg: {avg_reward:.2f}")
        return rewards
    
    # 准备训练提示
    def prepare_prompts(dataset: Dataset) -> List[Dict[str, str]]:
        prompts = []
        for item in dataset:
            prompt = generate_prediction_prompt(item)
            prompts.append({"prompt": prompt})
        return prompts
    
    train_prompts = prepare_prompts(train_data)
    val_prompts = prepare_prompts(val_data)
    
    # 加载基础模型
    logging.info(f"Loading base model from: {base_model_path}")
    
    # 配置量化 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 加载SFT阶段的LoRA权重
    logging.info(f"Loading SFT LoRA weights from: {sft_lora_weights}")
    model = PeftModel.from_pretrained(
        base_model,
        sft_lora_weights,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    
    # 特殊处理分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,  # 使用基础模型的分词器
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        # 优先尝试使用 EOS 标记作为填充标记
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        # 如果没有 EOS 标记，尝试使用 BOS 标记
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        # 如果都没有，添加一个新的特殊标记
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    
    # 添加新的LoRA层用于GRPO训练
    logging.info("Adding new LoRA layers for GRPO training")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    # 在现有模型上添加新的LoRA适配器
    model = get_peft_model(model, peft_config)
    
    # 冻结基础模型和SFT LoRA层，只训练新添加的LoRA层
    for name, param in model.named_parameters():
        if "lora" in name and "default" in name:  # 只解冻新添加的LoRA层
            param.requires_grad = True
            logging.debug(f"Training parameter: {name}")
        else:
            param.requires_grad = False
    
    # 记录可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params} | Total parameters: {total_params} | Percentage: {trainable_params/total_params*100:.4f}%")
    
    # 配置GRPO训练参数
    config = GRPOConfig(
        # 基础配置
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 每个设备上的提示数
        gradient_accumulation_steps=batch_size // 4,  # 梯度累积步数
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        seed=seed,
        fp16=True,  # 使用混合精度训练
        
        # GRPO特定参数
        num_generations=8,  # 每个提示生成8个补全
        temperature=0.9,    # 采样温度
        max_completion_length=128,  # 最大补全长度
        beta=0.04,          # KL系数
        remove_unused_columns=False,  # 保留所有列以便奖励计算
        
        # 生成参数
        max_prompt_length=cutoff_len,  # 最大提示长度
        
        # 日志和报告
        logging_steps=50,  # 减少日志频率
        report_to="tensorboard",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # 评估设置
        eval_strategy="no",        # 每个epoch结束后进行评估
        save_strategy="epoch",        # 每个epoch结束后保存模型
        save_total_limit=2,           # 最多保存2个检查点
    )
    
    # 创建GRPO训练器
    logging.info("Creating GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_prompts,  # 传入训练提示
        eval_dataset=val_prompts      # 添加验证集用于评估
    )
    trainer.model.config.use_cache = False
    # 确保训练器使用的分词器有正确的填充标记
    if trainer.tokenizer.pad_token is None:
        if trainer.tokenizer.eos_token is not None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token
        elif trainer.tokenizer.bos_token is not None:
            trainer.tokenizer.pad_token = trainer.tokenizer.bos_token
        else:
            trainer.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    trainer.tokenizer.padding_side = "left"
    
    # 添加自定义回调函数
    trainer.add_callback(SavePeftModelCallback())
    trainer.add_callback(EpochLoggingCallback())
    
    # 训练模型
    logging.info("Starting GRPO training...")
    start_time = datetime.datetime.now()
    trainer.train()
    training_duration = datetime.datetime.now() - start_time
    
    # 记录训练总时长
    logging.info(f"\n{'='*50}")
    logging.info(f"Training completed!")
    logging.info(f"Total Training Time: {training_duration}")
    logging.info(f"Average Time per Epoch: {training_duration / num_epochs}")
    logging.info(f"{'='*50}\n")
    
    # 保存最终模型
    save_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"GRPO training complete. Model saved to: {save_path}")
    
    # 评估模型
    avg_reward = evaluate_grpo_model(
        save_path,  # 直接使用保存的模型路径
        val_data, 
        tokenizer, 
        reward_calculator
    )
    
    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            "average_reward": avg_reward,
            "training_params": {
                "base_model": base_model_path,
                "sft_lora_weights": sft_lora_weights,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size
            },
            "training_duration": str(training_duration),
            "completion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

def evaluate_grpo_model(model_path, dataset, tokenizer, reward_calculator, num_samples=50):
    """评估GRPO模型性能"""
    logging.info("Evaluating GRPO model...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 选择随机样本进行评估
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    total_reward = 0
    rewards = []
    results = []
    
    for idx in sample_indices:
        data_point = dataset[idx]
        prompt = generate_prediction_prompt(data_point)
        
        # 生成推荐
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成部分 - 只处理Alpaca格式
        if "### Response:" in generated_text:
            generated_output = generated_text.split("### Response:")[-1].strip()
            if "###" in generated_output:
                generated_output = generated_output.split("###")[0].strip()
        else:
            generated_output = generated_text.strip()
        
        # 提取用户历史
        user_history = data_point.get("input", "")
        
        # 计算奖励
        reward = reward_calculator.calculate_reward(user_history, generated_output)
        rewards.append(reward)
        total_reward += reward
        
        # 保存结果
        result = {
            "user_history": user_history,
            "generated_recommendation": generated_output,
            "reward": reward
        }
        results.append(result)
        
        # 记录结果
        logging.info(f"\nSample {idx+1}/{num_samples}")
        logging.info(f"User History: {user_history}")
        logging.info(f"Generated Recommendation: {generated_output}")
        logging.info(f"Reward: {reward:.4f}")
        logging.info("-" * 50)
    
    avg_reward = total_reward / num_samples
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    logging.info(f"Evaluation Results:")
    logging.info(f"Average Reward: {avg_reward:.4f}")
    logging.info(f"Min Reward: {min_reward:.4f}")
    logging.info(f"Max Reward: {max_reward:.4f}")
    
    # 保存详细结果
    with open(os.path.join(model_path, "detailed_evaluation.json"), "w") as f:
        json.dump({
            "average_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "samples": results
        }, f, indent=2)
    
    return avg_reward

if __name__ == "__main__":
    fire.Fire(train_grpo)