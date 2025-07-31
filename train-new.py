import os
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
os.environ["WANDB_DISABLED"] = "true"
os.environ['NCCL_SOCKET_TIMEOUT'] = '7200'
os.environ['NCCL_DEBUG'] = 'WARN'
import sys
import json
import fire
import random
import torch
import logging
import datetime
import numpy as np
import transformers
from typing import List, Dict, Any
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerState, 
    TrainerControl,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 配置日志系统
def setup_logging(output_dir, model_type):
    # 创建logs目录
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成基于时间的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_type}_training_{timestamp}.log"
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

# 通用模板生成函数，兼容不同模型
def generate_prompt(data_point: Dict[str, Any], model_type: str = "llama") -> str:
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")
    
    # 特殊处理Qwen2的模板
    if model_type == "qwen":
        if input_text:
            return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        else:
            return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    # 默认Alpaca格式
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
{output}"""

# 预测时使用的模板生成
def generate_prediction_prompt(data_point: Dict[str, Any], model_type: str = "llama") -> str:
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    
    if model_type == "qwen":
        if input_text:
            return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    # 默认Alpaca格式
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

class SamplePredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_data, model_type="llama", device_map="auto"):
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.device_map = device_map
        self.model_type = model_type
        self.sample_index = random.randint(0, len(raw_data) - 1)
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.local_rank not in [-1, 0]:
            return
        if not self.raw_data:
            logging.warning("No test data available for sampling")
            return
            
        sample_data = self.raw_data[self.sample_index]
        prompt = generate_prediction_prompt(sample_data, self.model_type)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                num_beams=1,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 特殊处理Qwen2的输出格式
        if self.model_type == "qwen":
            if "<|im_start|>assistant" in decoded_output:
                predicted_output = decoded_output.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in predicted_output:
                    predicted_output = predicted_output.split("<|im_end|>")[0].strip()
        else:
            if "### Response:" in decoded_output:
                predicted_output = decoded_output.split("### Response:")[-1].strip()
                if "###" in predicted_output:
                    predicted_output = predicted_output.split("###")[0].strip()
            else:
                predicted_output = decoded_output.strip()
        
        log_msg = [
            "\n" + "="*80,
            f"Epoch {int(state.epoch)} - Sample Prediction",
            "="*80,
            f"Instruction:\n{sample_data['instruction']}"
        ]
        
        if sample_data.get("input", ""):
            log_msg.append(f"\nInput:\n{sample_data['input']}")
        
        log_msg.extend([
            f"\nActual Output:\n{sample_data['output']}",
            f"\nPredicted Output:\n{predicted_output}",
            "="*80 + "\n"
        ])
        
        logging.info("\n".join(log_msg))
        self.sample_index = random.randint(0, len(self.raw_data) - 1)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        # 记录模型保存信息
        logging.info(f"Saved adapter model to: {peft_model_path}")

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def train(
    base_model: str = "base_models/Qwen2.5-0.5B",
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    val_test_path: List[str] = ["data/game/dataset/processed/test_5000.json"],
    output_dir: str = "./Qwen2.5-0.5B-lora-alpaca-game-2",
    sample: int = 1024,
    seed: int = 1,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1024,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    model_type: str = "auto",  # auto/llama/qwen
):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动检测模型类型
    if model_type == "auto":
        if "qwen" in base_model.lower():
            model_type = "qwen"
            print("Auto-detected model type: Qwen")
        elif "llama" in base_model.lower():
            model_type = "llama"
            print("Auto-detected model type: Llama")
        else:
            model_type = "llama"
            print("Using default model type: Llama")
    else:
        print(f"Using specified model type: {model_type}")
    
    # 设置日志系统
    log_filepath = setup_logging(output_dir, model_type)
    print(f"Logging training process to: {log_filepath}")
    logging.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    params = {
        "base_model": base_model,
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "val_test_path": val_test_path,
        "sample": sample,
        "seed": seed,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "cutoff_len": cutoff_len,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "train_on_inputs": train_on_inputs,
        "group_by_length": group_by_length,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_watch": wandb_watch,
        "wandb_log_model": wandb_log_model,
        "resume_from_checkpoint": resume_from_checkpoint,
        "model_type": model_type,
    }
    
    logging.info("Training Alpaca-LoRA model with params:")
    for key, value in params.items():
        logging.info(f"{key}: {value}")
    
    # 记录到控制台
    print("\nTraining Alpaca-LoRA model with params:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print()
    
    assert base_model, "Please specify a --base_model"
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        # 分布式训练：使用LOCAL_RANK指定的设备
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    else:
        # 单GPU：使用当前活跃的GPU
        device_map = {"": torch.cuda.current_device()}
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        logging.info(f"Distributed Data Parallel enabled. World size: {world_size}")
    
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    logging.info(f"Loading model from: {base_model}")
    # 通用模型加载
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True
    )
    
    logging.info(f"Loading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 特殊处理分词器
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Qwen2需要设置特殊token
    if model_type == "qwen":
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.pad_token = "<|endoftext|>"
    
    logging.info(f"Tokenizer settings - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}, padding_side: {tokenizer.padding_side}")
    
    def tokenize(prompt: str, add_eos_token: bool = True) -> Dict[str, List[int]]:
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        # 添加EOS token
        if (
            len(result["input_ids"]) < cutoff_len and
            add_eos_token and
            result["input_ids"][-1] != tokenizer.eos_token_id
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point: Dict[str, Any]) -> Dict[str, List[int]]:
        full_prompt = generate_prompt(data_point, model_type)
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""}, model_type)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
        return tokenized_full_prompt

    # 准备模型训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    def load_data(paths: List[str], is_train: bool = False):
        datasets = []
        for path in paths:
            logging.info(f"Loading data from: {path}")
            if path.endswith(".json"):
                dataset = load_dataset("json", data_files=path)["train"]
            else:
                dataset = load_dataset(path)["train"]
            
            if is_train and sample > -1:
                logging.info(f"Sampling {sample} examples from dataset")
                dataset = dataset.shuffle(seed=seed).select(range(sample))
            elif is_train:
                logging.info("Shuffling full training dataset")
                dataset = dataset.shuffle(seed=seed)
            
            datasets.append(dataset)
        
        concatenated = concatenate_datasets(datasets)
        logging.info(f"Total samples after concatenation: {len(concatenated)}")
        return concatenated.map(generate_and_tokenize_prompt)

    train_data = load_data(train_data_path, is_train=True)
    val_data = load_data(val_data_path)
    
    # 加载原始测试数据用于预测展示
    raw_test_data = concatenate_datasets([
        load_dataset("json", data_files=path)["train"] if path.endswith(".json") 
        else load_dataset(path)["train"] 
        for path in val_test_path
    ])
    
    logging.info(f"Loaded training data: {len(train_data)} samples")
    logging.info(f"Loaded validation data: {len(val_data)} samples")
    logging.info(f"Loaded test data for prediction: {len(raw_test_data)} samples")
    
    # 恢复检查点
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model", "adapter_model.bin")
        
        if os.path.exists(checkpoint_name):
            logging.info(f"Restarting from checkpoint: {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logging.warning(f"Checkpoint not found: {checkpoint_name}")
    
    # 记录可训练参数信息
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params} | Total parameters: {total_params} | Percentage: {trainable_params/total_params*100:.4f}%")
    
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
        logging.info(f"Using model parallel on {torch.cuda.device_count()} GPUs")

    # 训练回调函数
    early_stopping = EarlyStoppingCallback(early_stopping_patience=15)
    sample_prediction = SamplePredictionCallback(
        tokenizer=tokenizer,
        raw_data=raw_test_data,
        model_type=model_type,
        device_map=device_map
    )
    
    # 训练参数配置
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs", "tensorboard"),
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        label_pad_token_id=-100
    )
    
    # 创建Trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
        callbacks=[early_stopping, sample_prediction, SavePeftModelCallback]
    )
    
    model.config.use_cache = False
    
    # 开始训练
    logging.info("Starting training...")
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存最终模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"\nTraining complete. Final model saved to: {output_dir}")
    print(f"\nTraining complete. Final model saved to: {output_dir}")
    
    # 记录训练结束时间
    logging.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total training time: {datetime.datetime.now() - start_time}")

if __name__ == "__main__":
    # 记录全局开始时间
    start_time = datetime.datetime.now()
    fire.Fire(train)