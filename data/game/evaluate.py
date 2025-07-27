from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch
import os
import math
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
        if name.endswith(".json"):
            path.append(os.path.join(root, name))
print(f"{path}")
base_model = "/home/lsx/code/BIGRec/base_models/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
device = model.device
print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(script_dir, 'dataset/id2name.txt'), 'r')
items = f.readlines()
item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
item_ids = [_ for _ in range(len(item_names))]
item_dict = dict()
for i in range(len(item_names)):
    if item_names[i] not in item_dict:
        item_dict[item_names[i]] = [item_ids[i]]
    else:   
        item_dict[item_names[i]].append(item_ids[i])
result_dict = dict()

movie_embedding_path = os.path.join(script_dir, 'dataset/embedding/item_embedding.pt')
movie_embedding = torch.load(movie_embedding_path, weights_only=True).to(device)
num_items = movie_embedding.size(0) 
print(f"Loaded movie embeddings with shape: {movie_embedding.shape}")
import pandas as pd

for p in path:
    if p.endswith("_evaluation.json"):
        continue
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()
    #text = [_["predict"][0].strip("\"") for _ in test_data]
    #text = [_["predict"][0].strip("\"") for _ in test_data if _["predict"] is not None]
    filtered_test_data = [_ for _ in test_data if _["predict"] is not None]
    text = [_["predict"][0].strip("\"") for _ in filtered_test_data]
    tokenizer.padding_side = "left"
    test_data = filtered_test_data
    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    for batch_input in tqdm(batch(text, 8), total=len(text)//8+1):
        inputs = tokenizer(
            batch_input, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=1024
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0)
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)

    rank = dist
    rank = rank.argsort(dim = -1).argsort(dim = -1)
    topk_list = [1, 3, 5, 10, 20, 50]
    NDCG = []
    HR = []
    for topk in topk_list:
        S = 0
        SS = 0
        LL = len(test_data)
        for i in range(len(test_data)):
            target_item = test_data[i]['output'].strip("\"").strip(" ")
            minID = 20000
            for _ in item_dict[target_item]:
                if _ < rank.shape[1]:
                    if rank[i][_].item() < minID:
                        minID = rank[i][_].item()
            if minID < topk:
                S= S+ (1 / math.log(minID + 2))
                SS = SS + 1
        temp_NDCG = []
        temp_HR = []
        NDCG.append(S / LL / (1.0 / math.log(2)))
        HR.append(SS / LL)
    
    print(f"File: {p}")
    print(f"NDCG@k: {NDCG}")
    print(f"HR@k: {HR}")
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR

f = open('data/game/result/game_evaluation.json', 'w')    
json.dump(result_dict, f, indent=4)