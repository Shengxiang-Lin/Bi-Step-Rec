import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
os.environ["WANDB_DISABLED"] = "true"
os.environ['NCCL_SOCKET_TIMEOUT'] = '7200'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" 
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
trl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trl"))
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)
from trl import GRPOTrainer, GRPOConfig
import torch.nn.functional as F

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

class RewardCalculator:
    def __init__(self):
        self.sentence_transformer_cache = "./sentence_transformer_cache"
        os.makedirs(self.sentence_transformer_cache, exist_ok=True)
        self.similarity_model = SentenceTransformer(
            'all-mpnet-base-v2',
            cache_folder=self.sentence_transformer_cache 
        )
        self.embedding_cache = {}
        self.w_relevance = 2.0
        self.w_diversity = 1.0
        self.w_novelty = 0.8
        self.ema_alpha = 0.9
        self.reward_history = []
        self.component_stds = {'relevance': [], 'diversity': [], 'novelty': []}
        self.penalty_repeat_base = 2.0
        self.penalty_irrelevant = 0.5
        self.game_freq = {}
        self.relevance_threshold = 0.3
        self.too_similar_threshold = 0.9
        self.adjustment_counter = 0
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        cache_key = (text1, text2)
        if cache_key in self.embedding_cache:
            similarity = self.embedding_cache[cache_key]
        else:
            embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
            similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
            similarity = (similarity + 1) / 2
            self.embedding_cache[cache_key] = similarity
        return similarity
    
    def extract_quoted_games(self, text: str) -> List[str]:
        games = re.findall(r'"([^"]+)"', text)
        games = [g for g in games if g.strip().lower() not in ['name of the recommended thing', '<name of the recommended thing>']]
        return games
    
    def update_weights(self, relevance: float, diversity: float, novelty: float):
        self.component_stds['relevance'].append(relevance)
        self.component_stds['diversity'].append(diversity)
        self.component_stds['novelty'].append(novelty)
        if len(self.component_stds['relevance']) > 100:
            self.component_stds['relevance'].pop(0)
            self.component_stds['diversity'].pop(0)
            self.component_stds['novelty'].pop(0)
        if len(self.component_stds['relevance']) >= 10:
            std_rel = np.std(self.component_stds['relevance']) + 1e-6
            std_div = np.std(self.component_stds['diversity']) + 1e-6
            std_nov = np.std(self.component_stds['novelty']) + 1e-6
            total_std = std_rel + std_div + std_nov
            new_w_rel = std_rel / total_std
            new_w_div = std_div / total_std
            new_w_nov = std_nov / total_std
            self.w_relevance = self.ema_alpha * self.w_relevance + (1 - self.ema_alpha) * new_w_rel
            self.w_diversity = self.ema_alpha * self.w_diversity + (1 - self.ema_alpha) * new_w_div
            self.w_novelty = self.ema_alpha * self.w_novelty + (1 - self.ema_alpha) * new_w_nov
            logging.info(f"Updated weights: relevance={self.w_relevance:.4f}, diversity={self.w_diversity:.4f}, novelty={self.w_novelty:.4f}")
    
    def calculate_reward(self, user_history: str, generated_output: str) -> float:
        history_games = self.extract_quoted_games(user_history)
        gen_games = self.extract_quoted_games(generated_output)
        gen_game = gen_games[0] if gen_games else generated_output.strip()
        logging.info(f"history_games: {history_games}")
        logging.info(f"gen_game: {gen_game}")
        gen_game_lower = gen_game.lower()
        self.game_freq[gen_game_lower] = self.game_freq.get(gen_game_lower, 0) + 1
        if any(gen_game_lower == game.lower() for game in history_games):
            penalty_repeat = self.penalty_repeat_base * (1 + np.log1p(self.game_freq[gen_game_lower]))
            total_reward = -penalty_repeat
            logging.info(f"Repeat penalty: -{penalty_repeat:.2f}")
            return total_reward
        relevance = self.compute_relevance(history_games, gen_game)
        diversity = self.compute_diversity(history_games, gen_game)
        novelty = self.compute_novelty(history_games, gen_game)
        self.update_weights(relevance, diversity, novelty)
        reward = self.w_relevance * relevance + self.w_diversity * diversity + self.w_novelty * novelty
        if relevance < self.relevance_threshold:
            reward -= self.penalty_irrelevant
            logging.debug(f"Irrelevant penalty: -{self.penalty_irrelevant}")
        elif novelty < 0.2:
            reward -= self.penalty_irrelevant * 0.5
            logging.debug(f"Too similar penalty: -{self.penalty_irrelevant*0.5}")
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        if len(self.reward_history) >= 10:
            median_reward = np.median(self.reward_history)
            iqr = np.percentile(self.reward_history, 75) - np.percentile(self.reward_history, 25)
            if iqr > 0:
                normalized_reward = (reward - median_reward) / (1.5 * iqr)
                total_reward = max(-2.0, min(2.0, normalized_reward))
            else:
                total_reward = reward
        else:
            total_reward = reward
        logging.info(f"Relevance: {relevance:.4f}, Diversity: {diversity:.4f}, Novelty: {novelty:.4f}")
        logging.info(f"Reward: {reward:.2f}, Normalized: {total_reward:.2f}")
        return total_reward
    
    def compute_relevance(self, history: List[str], gen_game: str) -> float:
        if not history:
            return 0.7
        weights = [0.9**i for i in range(len(history))]
        weights.reverse()
        total_weight = sum(weights)
        similarities = [self.compute_similarity(gen_game, game) for game in history]
        weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        return 1 / (1 + np.exp(-10 * (weighted_avg - 0.5)))
    
    def compute_diversity(self, history: List[str], gen_game: str) -> float:
        if len(history) < 2:
            return 0.5
        history_sims = []
        for i in range(len(history)):
            for j in range(i+1, len(history)):
                history_sims.append(self.compute_similarity(history[i], history[j]))
        hist_diversity = 1 - np.mean(history_sims) if history_sims else 0
        new_history = history + [gen_game]
        new_sims = []
        for i in range(len(new_history)):
            for j in range(i+1, len(new_history)):
                new_sims.append(self.compute_similarity(new_history[i], new_history[j]))
        new_diversity = 1 - np.mean(new_sims) if new_sims else 0
        diversity_gain = max(0, new_diversity - hist_diversity)
        return np.log1p(diversity_gain * 10) / 2.0
    
    def compute_novelty(self, history: List[str], gen_game: str) -> float:
        if not history:
            return 1.0
        min_sim = min(self.compute_similarity(gen_game, game) for game in history)
        novelty = 1 - min_sim
        if novelty < 0.3:
            return novelty * 0.5
        elif novelty > 0.7:
            return min(1.0, novelty * 1.2)
        return novelty

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

class SavePeftModelCallback(TrainerCallback):
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
            remaining_epochs = args.num_train_epochs - epoch_num
            estimated_remaining = avg_epoch_time * remaining_epochs
            logging.info(f"\n{'='*50}")
            logging.info(f"Completed Epoch {epoch_num}")
            logging.info(f"Epoch Duration: {epoch_duration}")
            logging.info(f"Average Epoch Time: {avg_epoch_time}")
            logging.info(f"Estimated Remaining Time: {estimated_remaining}")
            logging.info(f"{'='*50}\n")
        self.epoch_start_time = None

def train_grpo(
    load_8bit: bool = False,
    base_model_path: str = "base_models/Qwen2.5-3B-Instruct",
    sft_lora_weights: str = "./Qwen2.5-3B-Instruct-game-base-0/checkpoint-32",
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    output_dir: str = "./Qwen2.5-3B-Instruct-grpo-0-32",
    seed: int = 0,
    batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    cutoff_len: int = 256,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_sample: int = 10,
    val_sample: int = 1000 
):
    os.makedirs(output_dir, exist_ok=True)
    log_filepath = setup_logging(output_dir)
    print(f"Logging GRPO training process to: {log_filepath}")
    logging.info(f"GRPO training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using Qwen model: {base_model_path}")
    logging.info(f"Training Parameters: Epochs={num_epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
    reward_calculator = RewardCalculator()

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
    
    def reward_fn(prompts: List[str], completions: List[str]) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                match = re.search(r'<\|im_start\|>user\s*.*?<\|im_end\|>\s*(.*?)(<\|im_end\|>|$)', prompt, re.DOTALL)
                user_history = match.group(1).strip() if match else ""
                reward = reward_calculator.calculate_reward(user_history, completion)
                rewards.append(reward)
            except Exception as e:
                logging.error(f"Error calculating reward: {e}")
                rewards.append(0.0)
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            avg_reward = sum(rewards) / len(rewards)
            logging.info(f"Batch rewards - Min: {min_reward:.2f}, Max: {max_reward:.2f}, Avg: {avg_reward:.2f}")
        return rewards
    
    def prepare_prompts(dataset: Dataset) -> List[Dict[str, str]]:
        prompts = []
        for item in dataset:
            prompt = generate_prediction_prompt(item)
            prompts.append({"prompt": prompt})
        return prompts
    
    train_prompts = prepare_prompts(train_data)
    val_prompts = prepare_prompts(val_data)
    logging.info(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model = base_model
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
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "grpo_adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params} | Total parameters: {total_params} | Percentage: {trainable_params/total_params*100:.4f}%")
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
    logging.info("Creating GRPOTrainer")
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
    logging.info("Starting GRPO training...")
    start_time = datetime.datetime.now()
    trainer.train()
    training_duration = datetime.datetime.now() - start_time
    logging.info(f"\n{'='*50}")
    logging.info(f"Training completed!")
    logging.info(f"Total Training Time: {training_duration}")
    logging.info(f"Average Time per Epoch: {training_duration / num_epochs}")
    logging.info(f"{'='*50}\n")
    save_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"GRPO training complete. Model saved to: {save_path}")
    logging.info("Starting manual evaluation...")
    avg_reward = evaluate_grpo_model(
        model,
        tokenizer,
        val_data,
        reward_calculator
    )
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

def evaluate_grpo_model(model, tokenizer, dataset, reward_calculator, num_samples=50):
    logging.info("Evaluating GRPO model...")
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    total_reward = 0
    rewards = []
    results = []
    for idx in sample_indices:
        data_point = dataset[idx]
        prompt = generate_prediction_prompt(data_point)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
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
        user_history = data_point.get("input", "")
        reward = reward_calculator.calculate_reward(user_history, generated_output)
        rewards.append(reward)
        total_reward += reward
        result = {
            "user_history": user_history,
            "generated_recommendation": generated_output,
            "prompt": prompt,
            "reward": reward
        }
        results.append(result)
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

if __name__ == "__main__":
    fire.Fire(train_grpo)