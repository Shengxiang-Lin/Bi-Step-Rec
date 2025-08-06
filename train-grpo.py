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
        # 核心奖励权重
        self.w_relevance = 2.0    # 相关性奖励权重
        self.w_diversity = 1.0    # 多样性奖励权重
        self.w_novelty = 0.8      # 新颖性奖励权重
        # 惩罚权重
        self.penalty_repeat = 2.0 # 重复历史项惩罚
        self.penalty_irrelevant = 0.5  # 完全不相关惩罚
        # 阈值参数
        self.relevance_threshold = 0.3  # 完全不相关阈值
        self.too_similar_threshold = 0.9  # 过于相似阈值
        # 用于奖励归一化
        self.reward_history = []
        self.adjustment_counter = 0
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个游戏名称之间的余弦相似度"""
        embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
        return (similarity + 1) / 2  # 归一化到[0, 1]
    
    def extract_quoted_games(self, text: str) -> List[str]:
        """
        提取所有被双引号包裹的游戏名称，并过滤掉模板字段
        """
        games = re.findall(r'"([^"]+)"', text)
        games = [g for g in games if g.strip().lower() not in ['name of the recommended thing', '<name of the recommended thing>']]
        return games
    
    def calculate_reward(self, user_history: str, generated_output: str) -> float:
        """
        计算生成商品的奖励 - 基于相关性、多样性和新颖性
        """
        # 从用户历史中提取游戏名称
        history_games = self.extract_quoted_games(user_history)
        # 从生成的输出中提取游戏名称
        gen_games = self.extract_quoted_games(generated_output)
        gen_game = gen_games[0] if gen_games else generated_output.strip()
        logging.info(f"history_games: {history_games}")
        logging.info(f"gen_game: {gen_game}")
        # 1. 检查是否重复历史项（严重惩罚）
        if any(gen_game.lower() == game.lower() for game in history_games):
            total_reward = -self.penalty_repeat
            logging.info(f"Repeat penalty: -{self.penalty_repeat}")
            return total_reward
        # 2. 计算相关性奖励
        relevance = self.compute_relevance(history_games, gen_game)
        reward = self.w_relevance * relevance
        # 3. 计算多样性奖励
        diversity = self.compute_diversity(history_games, gen_game)
        reward += self.w_diversity * diversity
        # 4. 计算新颖性奖励
        novelty = self.compute_novelty(history_games, gen_game)
        reward += self.w_novelty * novelty
        # 5. 添加惩罚项（轻微）
        if relevance < self.relevance_threshold:  # 完全不相关
            reward -= self.penalty_irrelevant
            logging.debug(f"Irrelevant penalty: -{self.penalty_irrelevant}")
        elif novelty < 0.2:  # 与历史游戏过于相似
            reward -= self.penalty_irrelevant * 0.5
            logging.debug(f"Too similar penalty: -{self.penalty_irrelevant*0.5}")
        # 记录奖励历史用于归一化
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        # 归一化奖励（基于近期奖励分布）
        if len(self.reward_history) >= 10:
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history)
            if std_reward > 0:
                normalized_reward = (reward - mean_reward) / std_reward
                # 限制在[-2, 2]范围内
                total_reward = max(-2.0, min(2.0, normalized_reward))
            else:
                total_reward = reward
        else:
            total_reward = reward
        # 日志记录
        logging.info(f"Relevance: {relevance:.4f}, Diversity: {diversity:.4f}, Novelty: {novelty:.4f}")
        logging.info(f"Reward: {reward:.2f}, Normalized: {total_reward:.2f}")
        return total_reward
    
    def compute_relevance(self, history: List[str], gen_game: str) -> float:
        """计算生成游戏与用户历史的相关性"""
        if not history:
            return 0.7  # 默认相关性（无历史时）
        # 使用加权平均相似度（近期历史权重更高）
        weights = [0.9**i for i in range(len(history))]  # 指数衰减权重
        weights.reverse()  # 最近的项目权重最高
        total_weight = sum(weights)
        similarities = [self.compute_similarity(gen_game, game) for game in history]
        weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        # 应用S形函数增强相关性感知
        return 1 / (1 + np.exp(-10 * (weighted_avg - 0.5)))
    
    def compute_diversity(self, history: List[str], gen_game: str) -> float:
        """计算生成游戏带来的多样性增益"""
        if len(history) < 2:
            return 0.5  # 小样本时默认中等多样性
        # 计算历史集合的多样性
        history_sims = []
        for i in range(len(history)):
            for j in range(i+1, len(history)):
                history_sims.append(self.compute_similarity(history[i], history[j]))
        hist_diversity = 1 - np.mean(history_sims) if history_sims else 0
        # 计算加入新游戏后的多样性
        new_history = history + [gen_game]
        new_sims = []
        for i in range(len(new_history)):
            for j in range(i+1, len(new_history)):
                new_sims.append(self.compute_similarity(new_history[i], new_history[j]))
        new_diversity = 1 - np.mean(new_sims) if new_sims else 0
        # 多样性增益
        diversity_gain = max(0, new_diversity - hist_diversity)
        # 应用对数变换增强小增益感知
        return np.log1p(diversity_gain * 10) / 2.0
    
    def compute_novelty(self, history: List[str], gen_game: str) -> float:
        """计算生成游戏的新颖性（与历史最小相似度）"""
        if not history:
            return 1.0  # 无历史时完全新颖
        # 计算与所有历史游戏的最小相似度
        min_sim = min(self.compute_similarity(gen_game, game) for game in history)
        # 新颖性 = 1 - 最小相似度
        novelty = 1 - min_sim
        # 应用非线性变换（奖励中等新颖性）
        if novelty < 0.3:
            return novelty * 0.5  # 低新颖性折扣
        elif novelty > 0.7:
            return min(1.0, novelty * 1.2)  # 高新颖性奖励
        return novelty

# 使用新的提示模板格式
def generate_prompt(data_point: Dict[str, Any]) -> str:
    """生成完整提示模板（包含输入和输出）"""
    if data_point["input"]:
        return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{data_point["instruction"]}
Only output in the following format, including the end token:
"<name of the recommended thing>"<|im_end|>
{data_point["input"]}<|im_end|>
<|im_start|>assistant
{data_point["output"]}<|im_end|>"""
    else:
        return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{data_point["instruction"]}Only output in the following format, including the end token:
"<name of the recommended thing>"<|im_end|>
<|im_start|>assistant
{data_point["output"]}<|im_end|>"""

def generate_prediction_prompt(data_point: Dict[str, Any]) -> str:
    """生成预测提示模板（只包含输入）"""
    if data_point["input"]:
        return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{data_point["instruction"]}
Only output in the following format, including the end token:"<name of the recommended thing>"<|im_end|>
{data_point["input"]}<|im_end|>
<|im_start|>assistant
"""
    else:
        return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{data_point["instruction"]}Only output in the following format, including the end token:"<name of the recommended thing>"<|im_end|>
<|im_start|>assistant
"""

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
    load_8bit: bool = False,
    base_model_path: str = "base_models/Qwen2.5-3B-Instruct",  # 基础模型路径
    sft_lora_weights: str = None,#"./Qwen2.5-3B-Instruct-game-base-0/checkpoint-32",  # SFT阶段保存的LoRA权重
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    output_dir: str = "./Qwen2.5-3B-Instruct-grpo-0-0",
    seed: int = 0,
    batch_size: int = 8,
    num_epochs: int = 3,  # RL训练通常需要较少轮次
    learning_rate: float = 1e-6,
    cutoff_len: int = 256,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_sample: int = 1024,  # 训练集采样数量，-1表示使用全部
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
                # 提取用户历史：位于 prompt 中 <|im_start|>user 和 <|im_end|> 之间
                match = re.search(r'<name of the recommended thing>"<\|im_end\|>\s*(.*?)<\|im_end\|>', prompt, re.DOTALL)
                user_history = match.group(1).strip() if match else ""
                #logging.info(f"[DEBUG] Extracted user_history: {user_history}")
                #logging.info(f"[DEBUG] Completion: {completion}")
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
        bnb_4bit_quant_type="fp4",  # ✅ 改为 fp4 更稳定
        bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ 改为 bfloat16
        bnb_4bit_use_double_quant=False,        # ✅ 关闭 double quant
    )
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model = base_model
    if sft_lora_weights is not None:
    # 加载SFT阶段的LoRA权重
        logging.info(f"Loading SFT LoRA weights from: {sft_lora_weights}")
        model = PeftModel.from_pretrained(
            base_model,
            sft_lora_weights,
            torch_dtype=torch.float16,
            device_map="auto",
            is_trainable=False,
            local_files_only=True,
            adapter_name="sft_adapter",
        )
        model = model.merge_and_unload()
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
    model.add_adapter(peft_config, "grpo_adapter")
    # 在现有模型上添加新的LoRA适配器
    model.set_adapter("grpo_adapter")
    # 冻结基础模型和SFT LoRA层，只训练新添加的LoRA层
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "grpo_adapter" in name:
            param.requires_grad = True
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
        per_device_train_batch_size=2,  # 每个设备上的提示数
        gradient_accumulation_steps=batch_size // 4,  # 梯度累积步数
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        seed=seed,
        fp16=True,  # 使用混合精度训练
        # GRPO特定参数
        num_generations=4,  # 每个提示生成4个补全
        temperature=0.7,  # 采样温度
        max_completion_length=64,  # 最大补全长度
        beta=0.01,          # KL系数
        remove_unused_columns=False,  # 保留所有列以便奖励计算
        # 生成参数
        max_prompt_length=cutoff_len,  # 最大提示长度
        # 日志和报告
        logging_steps=50,  # 减少日志频率
        report_to="tensorboard",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 评估设置
        eval_strategy="no",           # 每个epoch结束后进行评估
        save_strategy="epoch",        # 每个epoch结束后保存模型
        save_total_limit=5,           # 最多保存2个检查点
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
            max_new_tokens=64,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取生成部分 - 只处理Alpaca格式
        if "assistant" in generated_text:
            generated_output = generated_text.split("assistant")[-1].strip()
            if "<|im_end|>" in generated_output:
                generated_output = generated_output.split("<|im_end|>")[0].strip()
            if "<|endoftext|>" in generated_output:
                generated_output = generated_output.split("<|endoftext|>")[0].strip()
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