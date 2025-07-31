from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import re  

base_model = "../../base_models/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()

text = []
with open('dataset/movies.dat', 'r', encoding='ISO-8859-1') as f:
    for line in f:
        # Split each line of data, format：movie_id::title (year)::genres
        parts = line.strip().split('::')
        if len(parts) < 3:
            continue
        movie_title = parts[1].strip()
        text.append(movie_title)
tokenizer.padding_side = "left"

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
    hidden_states = outputs.hidden_states
    item_embedding.append(hidden_states[-1][:, -1, :].detach().cpu())

item_embedding = torch.cat(item_embedding, dim=0)
os.makedirs('dataset/embedding', exist_ok=True)
torch.save(item_embedding, 'dataset/embedding/movie_embedding-llama-7b.pt')
print(f"Shape of the generated embedding matrix: {item_embedding.shape}")