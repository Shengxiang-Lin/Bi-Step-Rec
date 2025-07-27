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
import re
from transformers.utils import logging
logging.set_verbosity_error()

from peft import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def main(
    load_8bit: bool = False,
    base_model: str = "base_models/llama-7b",
    lora_weights: str = "lora-alpaca-game/checkpoint-40",
    test_data_path: str = "data/game/dataset/processed/test_5000.json",
    result_json_data: str = "data/game/result/game.json",
    batch_size: int = 8,
):
    assert base_model, "Please specify a --base_model"

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
    tokenizer.pad_token_id = 0  # unk token
    model.config.pad_token_id = tokenizer.pad_token_id
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
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        prompts = [generate_prompt(inst, inp) for inst, inp in zip(instructions, inputs)]
        encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True if temperature > 0 else False,
            **kwargs,
        )

        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [out.split("### Response:\n")[-1].strip() for out in decoded]

        def extract_response(text):
            quoted_match = re.search(r'"([^"]+)"', text)
            if quoted_match:
                return f'"{quoted_match.group(1)}"'
            unquoted_match = re.search(r'([A-Z][a-zA-Z0-9 :\-\',&]+\(\d{4}\))', text)
            if unquoted_match:
                return f'"{unquoted_match.group(0)}"'
            title_match = re.search(r'([A-Z][a-zA-Z0-9 :\-\',&]+)', text)
            if title_match:
                return f'"{title_match.group(0)}"'
            cleaned_text = ' '.join(text.split())
            return f'"{cleaned_text[:100]}..."' if len(cleaned_text) > 100 else f'"{cleaned_text}"'
        
        predicted_response = [extract_response(text) for text in cleaned]
        return predicted_response

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    instructions = [item['instruction'] for item in test_data]
    inputs = [item['input'] for item in test_data]
    outputs = []

    with tqdm(total=len(instructions), desc="Generating") as pbar:
        for i in range(0, len(instructions), batch_size):
            batch_instructions = instructions[i:i+batch_size]
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = evaluate(batch_instructions, batch_inputs)
            outputs.extend(batch_outputs)
            pbar.update(len(batch_instructions))

    for i, item in enumerate(test_data):
        item['predict'] = outputs[i]

    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

if __name__ == "__main__":
    fire.Fire(main)
