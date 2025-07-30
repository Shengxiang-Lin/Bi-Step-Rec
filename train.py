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
import numpy as np
import transformers
from typing import List
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    EarlyStoppingCallback,
    TrainerCallback,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerState, 
    TrainerControl,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def generate_prompt(data_point):
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

class SamplePredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, raw_data, device_map="auto"):
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.device_map = device_map
        self.sample_index = random.randint(0, len(raw_data) - 1)
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.local_rank not in [-1, 0]:
            return
        if not self.raw_data:
            print("Warning: No test data available for sampling")
            return
        sample_data = self.raw_data[self.sample_index]
        if sample_data["input"]:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample_data["instruction"]}

### Input:
{sample_data["input"]}

### Response:"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample_data["instruction"]}

### Response:"""
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
                temperature=0.15,
                top_p=0.9,
                num_beams=1,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in decoded_output:
            predicted_output = decoded_output.split("### Response:")[-1].strip()
        else:
            predicted_output = decoded_output.strip()
        print("\n" + "="*80)
        print(f"Epoch {int(state.epoch)} - Sample Prediction")
        print("="*80)
        print(f"Instruction:\n{sample_data['instruction']}")
        if sample_data["input"]:
            print(f"\nInput:\n{sample_data['input']}")
        print(f"\nActual Output:\n{sample_data['output']}")
        print(f"\nPredicted Output:\n{predicted_output}")
        print("="*80 + "\n")
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

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def train(
    base_model: str = "base_models/Qwen2-0.5B",
    train_data_path: List[str] = ["data/game/dataset/processed/train.json"],
    val_data_path: List[str] = ["data/game/dataset/processed/valid_5000.json"],
    val_test_path: List[str] = ["data/game/dataset/processed/test_5000.json"],
    output_dir: str = "./Qwen2-0.5B-lora-alpaca-game",
    sample: int = 1024,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
):
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
    }
    print("Training Alpaca-LoRA model with params:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print()
    assert base_model, "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = "auto"
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        local_files_only=True
    )
    #train with llama-7b
    #tokenizer = LlamaTokenizer.from_pretrained(base_model, local_files_only=True)
    #trained with qwen2-0.5b
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id and
            len(result["input_ids"]) < cutoff_len and
            add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    def load_data(paths, is_train=False):
        datasets = []
        for path in paths:
            if path.endswith(".json"):
                dataset = load_dataset("json", data_files=path)["train"]
            else:
                dataset = load_dataset(path)["train"]
            if is_train and sample > -1:
                dataset = dataset.shuffle(seed=seed).select(range(sample))
            elif is_train:
                dataset = dataset.shuffle(seed=seed)
            datasets.append(dataset)
        concatenated = concatenate_datasets(datasets)
        return concatenated.map(generate_and_tokenize_prompt)

    train_data = load_data(train_data_path, is_train=True)
    val_data = load_data(val_data_path)
    raw_test_data = concatenate_datasets([
        load_dataset("json", data_files=path)["train"] if path.endswith(".json") 
        else load_dataset(path)["train"] 
        for path in val_test_path
    ])
    print(f"Loaded test data with {len(raw_test_data)} samples")
    prediction_dir = 'data/game/result'
    os.makedirs(prediction_dir, exist_ok=True)
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    model.print_trainable_parameters()
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    sample_prediction = SamplePredictionCallback(
        tokenizer=tokenizer,
        raw_data=raw_test_data,
        device_map=device_map
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20, 
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=50, 
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,
        greater_is_better=False, 
        metric_for_best_model="eval_loss", 
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[early_stopping, sample_prediction, SavePeftModelCallback]
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\nTraining complete. Final model saved.")

if __name__ == "__main__":
    fire.Fire(train)