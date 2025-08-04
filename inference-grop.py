import sys
import fire
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    load_8bit: bool = False,
    base_model: str = "base_models/llama-7b",
    sft_lora_weights: str = "llama-7b-lora-alpaca-game-base-0/checkpoint-40",
    grpo_lora_weights: str = "llama-7b-grpo-recommender/final_model",
    test_data_path: str = "data/game/dataset/processed/test_5000.json",
    result_json_data: str = "data/game/result/grpo/llama-7b-grpo-final.json",
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_beams: int = 1
):
    assert base_model, "Please specify a --base_model"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 特殊处理分词器
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"Loading base model: {base_model}")
    # 加载基础模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    # 加载SFT阶段的LoRA权重
    print(f"Loading SFT LoRA weights: {sft_lora_weights}")
    model = PeftModel.from_pretrained(
        model,
        sft_lora_weights,
        torch_dtype=torch.float16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else {"": device},
        local_files_only=True
    )
    
    # 加载GRPO阶段的LoRA权重
    print(f"Loading GRPO LoRA weights: {grpo_lora_weights}")
    model.load_adapter(grpo_lora_weights, adapter_name="grpo")
    
    # 激活GRPO适配器
    model.set_adapter("grpo")
    
    # 确保模型配置正确
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if not load_8bit and device == "cuda":
        model.half()  # 半精度可以减少内存使用
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def generate_prompt(instruction, input_text=None):
        """生成Llama格式的提示"""
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

    def extract_response(output: str) -> str:
        """从模型输出中提取响应部分"""
        if "### Response:" in output:
            response = output.split("### Response:")[-1].strip()
            # 移除可能的多余部分
            if "###" in response:
                response = response.split("###")[0].strip()
            return response
        return output.strip()

    def batch_evaluate(instructions, inputs):
        """批量评估函数"""
        prompts = []
        for instruction, input_text in zip(instructions, inputs):
            prompts.append(generate_prompt(instruction, input_text))
        
        # 编码输入
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        ).to(device)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        sequences = generation_output.sequences
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        # 提取响应部分
        return [extract_response(output) for output in outputs]

    print("Starting evaluation...")
    outputs = []
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [item['instruction'] for item in test_data]
        inputs = [item['input'] for item in test_data]
        
        total_samples = len(instructions)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(total=total_samples, desc="Evaluating")
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_instructions = instructions[i:batch_end]
            batch_inputs = inputs[i:batch_end]
            
            batch_outputs = batch_evaluate(batch_instructions, batch_inputs)
            outputs.extend(batch_outputs)
            
            # 更新进度条
            progress_bar.update(len(batch_outputs))
        
        progress_bar.close()
        
        # 将预测结果添加到测试数据中
        for i, item in enumerate(test_data):
            item['predict'] = outputs[i]

    # 确保结果目录存在
    result_dir = os.path.dirname(result_json_data)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    # 保存结果
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"\nEvaluation complete. Results saved to: {result_json_data}")
    print(f"Total samples evaluated: {len(test_data)}")

if __name__ == "__main__":
    fire.Fire(main)