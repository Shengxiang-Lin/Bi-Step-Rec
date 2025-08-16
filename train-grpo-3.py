import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import sys
import json
import fire
import random
import logging
import datetime
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
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
import re
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 添加本地 TRL 路径到系统路径（假设TRL库在本地）
trl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trl"))
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)
from trl import GRPOTrainer, GRPOConfig

# 配置环境变量
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
os.environ["WANDB_DISABLED"] = "true"
os.environ['NCCL_SOCKET_TIMEOUT'] = '7200'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" 

# ### 日志配置
def setup_logging(output_dir):
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"qwen_grpo_training_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_filepath

# ### Llama嵌入计算类
class LlamaEmbeddingCalculator:
    def __init__(self, model_path: str, batch_size: int = 8, max_length: int = 1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        logging.info(f"Loading Llama model from: {model_path}")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)
        self.embedding_cache = {}
        logging.info("Llama embedding model loaded successfully")
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        cached_embeddings = []
        need_compute = []
        need_compute_idx = []
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[text])
            else:
                need_compute.append(text)
                need_compute_idx.append(i)
        if need_compute:
            embeddings = []
            for i in range(0, len(need_compute), self.batch_size):
                batch_texts = need_compute[i:i+self.batch_size]
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                hidden_states = outputs.hidden_states[-1]
                batch_embeddings = hidden_states[torch.arange(hidden_states.size(0)), -1, :].detach().cpu()
                embeddings.append(batch_embeddings)
            all_embeddings = torch.cat(embeddings, dim=0)
            for j, text in enumerate(need_compute[i:i+self.batch_size]):
                self.embedding_cache[need_compute[i+j]] = all_embeddings[j]
        final_embeddings = []
        compute_idx = 0
        for i in range(len(texts)):
            if i in need_compute_idx:
                final_embeddings.append(all_embeddings[compute_idx])
                compute_idx += 1
            else:
                final_embeddings.append(cached_embeddings.pop(0))
        return torch.stack(final_embeddings).to(self.device)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.get_embeddings([text1, text2])
        emb1 = embeddings[0].unsqueeze(0)
        emb2 = embeddings[1].unsqueeze(0)
        similarity = F.cosine_similarity(emb1, emb2).item()
        normalized_similarity = (similarity + 1) / 2  # 归一化到 [0, 1]
        return normalized_similarity

# ### 奖励计算类
class RewardCalculator:
    def __init__(self, llama_model_path: str):
        self.embedding_calculator = LlamaEmbeddingCalculator(llama_model_path)
    
    def extract_quoted_content(self, text: str) -> str:
        matches = re.findall(r'"([^"]+)"', text)
        return matches[0] if matches else text.strip()
    
    def calculate_reward(self, generated_output: str, reference_output: str, user_history: List[str] = None) -> float:
        gen_content = self.extract_quoted_content(generated_output)
        ref_content = self.extract_quoted_content(reference_output)
        logging.info(f"User history: {user_history}")
        logging.info(f"Reference content: {ref_content}")
        logging.info(f"Generated content: {gen_content}")
        
        # 检查是否在用户历史记录中
        if user_history and gen_content in user_history:
            logging.info(f"Generated recommendations '{gen_content}' are penalized in the user's history.")
        #    return -1.0
        
        # 计算相似度作为奖励
        similarity = self.embedding_calculator.compute_similarity(gen_content, ref_content)
        return similarity

# ### 提示生成函数
def generate_prompt(data_point: Dict[str, Any]) -> str:
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

# ### 回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        logging.info(f"Saved adapter model to: {peft_model_path}")

class EpochLoggingCallback(TrainerCallback):
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
            remaining_epochs = args.num_train_epochs - epoch_num - 1
            estimated_remaining = avg_epoch_time * remaining_epochs
            logging.info(f"\n{'='*50}")
            logging.info(f"Completed Epoch {epoch_num}")
            logging.info(f"Epoch Duration: {epoch_duration}")
            logging.info(f"Average Epoch Time: {avg_epoch_time}")
            logging.info(f"Estimated Remaining Time: {estimated_remaining}")
            logging.info(f"{'='*50}\n")
        self.epoch_start_time = None

# ### GRPO训练函数
def train_grpo(
    load_8bit: bool = False,
    base_model_path: str = "base_models/Qwen2.5-0.5B-Instruct",
    sft_lora_weights: str = None,
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    output_dir: str = "./Qwen2.5-0.5B-Instruct-grpo-0-0-2",
    seed: int = 0,
    batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    cutoff_len: int = 256,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_sample: int = 1024,
    val_sample: int = 1024,
    llama_model_path: str = "/home/lsx/code/Bi-Step-Rec/base_models/llama-7b"
):
    os.makedirs(output_dir, exist_ok=True)
    log_filepath = setup_logging(output_dir)
    print(f"Logging GRPO training process to: {log_filepath}")
    logging.info(f"GRPO training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using Qwen model: {base_model_path}")
    logging.info(f"Using Llama model for reward: {llama_model_path}")
    logging.info(f"Training Parameters: Epochs={num_epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
    
    reward_calculator = RewardCalculator(llama_model_path)
    
    # 加载数据
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
        if is_train and sample > -1:
            logging.info(f"Sampling {sample} examples from training dataset")
            concatenated = concatenated.shuffle(seed=seed).select(range(sample))
        elif is_train:
            logging.info("Shuffling full training dataset")
            concatenated = concatenated.shuffle(seed=seed)
        elif sample > -1:
            logging.info(f"Sampling {sample} examples from validation dataset")
            concatenated = concatenated.shuffle(seed=seed).select(range(sample))
        logging.info(f"Final samples: {len(concatenated)}")
        return concatenated
    
    train_data = load_data(train_data_path, is_train=True, sample=train_sample)
    val_data = load_data(val_data_path, sample=val_sample)
    
    # 准备提示
    def prepare_prompts(dataset: Dataset) -> List[Dict[str, Any]]:
        prompts = []
        for item in dataset:
            prompt = generate_prediction_prompt(item)
            user_history = re.findall(r'"([^"]+)"', item["input"]) if item["input"] else []
            prompts.append({
                "prompt": prompt,
                "reference": item["output"],
                "user_history": user_history
            })
        return prompts
    
    train_prompts = prepare_prompts(train_data)
    val_prompts = prepare_prompts(val_data)
    
    # 加载基础模型
    logging.info(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model = base_model
    
    # 加载SFT LoRA权重（可选）
    if sft_lora_weights is not None:
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
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 准备模型进行LoRA微调
    model = prepare_model_for_kbit_training(model)
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
    model.set_adapter("grpo_adapter")
    
    # 设置可训练参数
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "grpo_adapter" in name:
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params} | Total parameters: {total_params} | Percentage: {trainable_params/total_params*100:.4f}%")
    
    # 奖励函数
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Calculate rewards for a batch of prompts and completions.

        Args:
            prompts (List[str]): List of prompt strings.
            completions (List[str]): List of completion strings.
            **kwargs: Additional arguments, including 'reference' for reward calculation.

        Returns:
            List[float]: List of reward scores.
        """
        # Retrieve reference outputs from kwargs
        reference = kwargs.get("reference", [])
        if not reference:
            raise ValueError("Reference outputs are required for reward calculation.")

        rewards = []
        for prompt, completion, ref in zip(prompts, completions, reference):
            try:
                # Example: Parse user history from prompt string (adjust based on actual format)
                # Assuming prompt format like: "...用户历史记录: \"game1\", \"game2\"..."
                user_history_match = re.search(r'The user has played the following video games before:(.*?)<\|im_end\|>', prompt, re.DOTALL)
                if user_history_match:
                    user_history_str = user_history_match.group(1)
                    user_history = re.findall(r'"([^"]+)"', user_history_str)
                else:
                    user_history = []
                #logging.info(f"Prompt: {prompt}")
                #logging.info(f"User history: {user_history}")
                # Calculate reward (assuming reward_calculator is defined elsewhere)
                reward = reward_calculator.calculate_reward(completion, ref, user_history)
                rewards.append(reward)
            except Exception as e:
                logging.error(f"Error calculating reward: {e}")
                rewards.append(0.0)  # Fallback reward value

        # Log reward statistics
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            avg_reward = sum(rewards) / len(rewards)
            logging.info(f"Batch rewards - Min: {min_reward:.4f}, Max: {max_reward:.4f}, Avg: {avg_reward:.4f}")

        return rewards
    
    # 配置GRPO训练参数
    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=batch_size // 4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        seed=seed,
        fp16=True,
        lr_scheduler_type="cosine",
        num_generations=4,
        temperature=0.7,
        max_completion_length=64,
        beta=0.01,
        remove_unused_columns=False,
        max_prompt_length=cutoff_len,
        logging_steps=50,
        report_to="tensorboard",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=5,
    )
    
    # 初始化GRPO训练器
    logging.info("Creating GRPOTrainer with reference-based rewards")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_prompts,
        eval_dataset=val_prompts
    )
    trainer.model.config.use_cache = False
    if trainer.tokenizer.pad_token is None:
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token
    trainer.tokenizer.padding_side = "left"
    
    trainer.add_callback(SavePeftModelCallback())
    trainer.add_callback(EpochLoggingCallback())
    
    # 开始训练
    logging.info("Starting GRPO training...")
    start_time = datetime.datetime.now()
    trainer.train()
    training_duration = datetime.datetime.now() - start_time
    logging.info(f"\n{'='*50}")
    logging.info(f"Training completed!")
    logging.info(f"Total Training Time: {training_duration}")
    logging.info(f"Average Time per Epoch: {training_duration / num_epochs}")
    logging.info(f"{'='*50}\n")
    
    # 保存模型
    save_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"GRPO training complete. Model saved to: {save_path}")
    
    # 手动评估
    logging.info("Starting manual evaluation...")
    avg_reward = evaluate_grpo_model(model, tokenizer, val_data, reward_calculator)
    
    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            "average_reward": avg_reward,
            "training_params": {
                "base_model": base_model_path,
                "sft_lora_weights": sft_lora_weights,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "llama_model_path": llama_model_path
            },
            "training_duration": str(training_duration),
            "completion_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

# ### 模型评估函数
def evaluate_grpo_model(model, tokenizer, dataset, reward_calculator, num_samples=50):
    logging.info("Evaluating GRPO model...")
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    total_reward = 0
    rewards = []
    results = []
    device = model.device
    
    for idx in sample_indices:
        data_point = dataset[idx]
        prompt = generate_prediction_prompt(data_point)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in generated_text:
            generated_output = generated_text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in generated_output:
                generated_output = generated_output.split("<|im_end|>")[0].strip()
        else:
            generated_output = generated_text.strip()
        
        user_history = re.findall(r'"([^"]+)"', data_point["input"]) if data_point["input"] else []
        reference_output = data_point["output"]
        reward = reward_calculator.calculate_reward(generated_output, reference_output, user_history)
        rewards.append(reward)
        total_reward += reward
        
        result = {
            "user_history": user_history,
            "generated_output": generated_output,
            "reference_output": reference_output,
            "prompt": prompt,
            "reward": reward
        }
        results.append(result)
        logging.info(f"\nSample {idx+1}/{num_samples}")
        logging.info(f"Generated Output: {generated_output}")
        logging.info(f"Reference Output: {reference_output}")
        logging.info(f"Similarity Reward: {reward:.4f}")
        logging.info("-" * 50)
    
    avg_reward = total_reward / num_samples
    min_reward = min(rewards)
    max_reward = max(rewards)
    logging.info(f"Evaluation Results:")
    logging.info(f"Average Reward: {avg_reward:.4f}")
    logging.info(f"Min Reward: {min_reward:.4f}")
    logging.info(f"Max Reward: {max_reward:.4f}")
    
    eval_dir = os.path.join(model.config._name_or_path, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, "detailed_evaluation.json")
    
    with open(eval_path, "w") as f:
        json.dump({
            "average_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "samples": results
        }, f, indent=2)
    
    logging.info(f"Detailed evaluation saved to: {eval_path}")
    return avg_reward

# ### 主函数
if __name__ == "__main__":
    fire.Fire(train_grpo)