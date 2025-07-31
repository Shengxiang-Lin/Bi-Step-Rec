import sys
import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "base_models/llama-7b",
    lora_weights: str = "llama-7b-lora-alpaca-game-2/checkpoint-112",
    test_data_path: str = "data/game/dataset/processed/test_5000.json",
    result_json_data: str = "data/game/result/num_beams/llama-7b-lora-alpaca-game-2.json",
    batch_size: int = 16,
    model_type: str = "auto",  # auto/llama/qwen
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='base_models/llama-7b'"
    
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
    
    # Qwen2需要设置特殊token
    if model_type == "qwen":
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.pad_token = "<|endoftext|>"
    
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
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    
    # 确保模型配置正确
    if model_type == "qwen":
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
    else:
        # 对于Llama模型，设置特殊token
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        do_sample=False,
        max_new_tokens=32,
        **kwargs,
    ):
        # 根据模型类型生成提示
        prompts = []
        for instruction, input_text in zip(instructions, inputs or [None]*len(instructions)):
            if model_type == "qwen":
                if input_text:
                    prompt = f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
"""
                else:
                    prompt = f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
            else:
                if input_text:
                    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
                else:
                    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:"""
            prompts.append(prompt)
        
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
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            num_return_sequences=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=False)
        
        # 根据模型类型提取响应
        real_outputs = []
        for output in outputs:
            if model_type == "qwen":
                if "<|im_start|>assistant" in output:
                    response = output.split("<|im_start|>assistant")[-1]
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0].strip()
                    else:
                        # 如果没有结束标记，保留整个响应
                        response = response.strip()
                else:
                    # 如果格式不匹配，返回整个输出
                    response = output.strip()
            else:
                if "### Response:" in output:
                    response = output.split("### Response:")[-1].strip()
                    if "###" in response:
                        response = response.split("###")[0].strip()
                else:
                    # 如果格式不匹配，返回整个输出
                    response = output.strip()
            real_outputs.append(response)
        
        # 分组为每个输入的多个输出
        grouped_outputs = [real_outputs[i * num_beams: (i + 1) * num_beams] 
                           for i in range(len(real_outputs) // num_beams)]
        return grouped_outputs

    outputs = []
    from tqdm import tqdm
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        
        def batch(list, batch_size=batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        
        # 分批处理
        for i, (batch_instructions, batch_inputs) in tqdm(
            enumerate(zip(batch(instructions), batch(inputs))), 
            total=(len(instructions)-1)//batch_size+1
        ):
            batch_outputs = evaluate(batch_instructions, batch_inputs)
            outputs.extend(batch_outputs)
        
        # 将预测结果添加到测试数据中
        for i, test in enumerate(test_data):
            test_data[i]['predict'] = outputs[i]

    # 确保结果目录存在
    result_dir = os.path.dirname(result_json_data)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    # 保存结果
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Evaluation complete. Results saved to: {result_json_data}")

if __name__ == "__main__":
    fire.Fire(main)