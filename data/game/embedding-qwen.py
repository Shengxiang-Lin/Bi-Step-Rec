from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os

base_model = "../../base_models/Qwen2.5-7B" 
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    padding_side="left",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True 
)
# 设置特殊token ID - Qwen使用<|endoftext|>作为pad/bos/eos
tokenizer.eos_token = "<|im_end|>"
tokenizer.bos_token = "<|im_start|>"
tokenizer.pad_token = tokenizer.eos_token  # 用eos_token作为pad_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
f = open('dataset/id2name.txt', 'r')
lines = f.readlines()
f.close()
text = [_.split('\t')[0].strip(" ").strip('\"') for _ in lines] 

from tqdm import tqdm
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

item_embedding = []
for i, batch_input in tqdm(enumerate(batch(text, 16))):
    input = tokenizer(batch_input, return_tensors="pt", padding=True)
    input_ids = input.input_ids.to(model.device)
    attention_mask = input.attention_mask.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # 获取最后一层隐藏状态
    hidden_states = outputs.hidden_states[-1]  
    # 提取每个序列的最后一个有效token的隐藏状态
    last_token_indices = attention_mask.sum(dim=1) - 1
    last_token_embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
    item_embedding.append(last_token_embeddings.detach().cpu())

item_embedding = torch.cat(item_embedding, dim=0)
os.makedirs('dataset/embedding', exist_ok=True)
torch.save(item_embedding, 'dataset/embedding/item_embedding-Qwen2.5-7B.pt')
print(f"Embedding shape: {item_embedding.shape}")