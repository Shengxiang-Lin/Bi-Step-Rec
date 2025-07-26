from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os

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
f = open('dataset/id2name.txt', 'r')
# the format of the item name file is 
# item_name item_id
# A 0
# B 1
lines = f.readlines()
f.close()
text = [_.split('\t')[0].strip(" ").strip('\"') for _ in lines] # remove the leading and trailing spaces and quotess make sure this preprocess is the same as the prediction
tokenizer.padding_side = "left"

from tqdm import tqdm
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]
item_embedding = []
for i, batch_input in tqdm(enumerate(batch(text, 16))):
    input = tokenizer(batch_input, return_tensors="pt", padding=True)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    item_embedding.append(hidden_states[-1][:, -1, :].detach().cpu())

item_embedding = torch.cat(item_embedding, dim=0)
os.makedirs('dataset/embedding', exist_ok=True)
torch.save(item_embedding, 'dataset/embedding/item_embedding.pt')
print(item_embedding.shape)