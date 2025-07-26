import sys
import fire
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers.utils import logging
logging.set_verbosity_error()

from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer
from transformers import LlamaForCausalLM
from tqdm import tqdm

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
    lora_weights: str = "lora-alpaca-movie/checkpoint-32",
    test_data_path: str = "data/movie/dataset/processed/valid_50.json",
    result_json_data: str = "data/movie/result/movie.json",
    batch_size: int = 2,
):
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model, local_files_only=True)
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    if not load_8bit:
        model.half()
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
        max_new_tokens=512,
        **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
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
        sequences = generation_output.sequences
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        outputs = [output.split('Response:\n')[-1] for output in outputs]
        return [outputs[i * num_beams: (i + 1) * num_beams] for i in range(len(instructions))]

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    instructions = [item['instruction'] for item in test_data]
    inputs = [item['input'] for item in test_data]
    outputs = []
    total_samples = len(instructions)
    with tqdm(total=total_samples, desc="Processing", unit="sample") as pbar:
        for i in range(0, total_samples, batch_size):
            batch_instructions = instructions[i:i+batch_size]
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = evaluate(
                batch_instructions, 
                batch_inputs
            )
            outputs.extend(batch_outputs)
            pbar.update(len(batch_instructions))
    for i, item in enumerate(test_data):
        item['predict'] = outputs[i]
    
    result_dir = os.path.dirname(result_json_data)
    os.makedirs(result_dir, exist_ok=True)
    
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""

if __name__ == "__main__":
    fire.Fire(main)