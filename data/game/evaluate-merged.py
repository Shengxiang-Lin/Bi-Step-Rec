from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
import math
import json
import argparse

# 设置参数解析
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir", type=str, default="./", help="your model directory")
parse.add_argument("--predict_weight", type=float, default=0.5, help="weight for predict ranking")
parse.add_argument("--keywords_weight", type=float, default=0.5, help="weight for keywords ranking")
args = parse.parse_args()

# 获取所有JSON文件路径
path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
        if name.endswith(".json"):
            path.append(os.path.join(root, name))
print(f"Found {len(path)} JSON files: {path}")

# 基础模型设置
base_model = "base_models/llama-7b"

# 加载分词器和模型
if "qwen" in base_model.lower():
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        pad_token='<|endoftext|>' if "qwen" in base_model.lower() else None
    )
else:  # Llama模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)

# 特殊处理pad token
if tokenizer.pad_token is None:
    if hasattr(tokenizer, 'eod_id'):  # Qwen的特殊字段
        tokenizer.pad_token = tokenizer.eod
    else:
        tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True if "qwen" in base_model.lower() else None
)

device = model.device
print(f"Using device: {device}")

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载物品字典
id2name_path = os.path.join(script_dir, 'dataset/id2name.txt')
if not os.path.exists(id2name_path):
    print(f"Error: id2name.txt not found at {id2name_path}")
    exit(1)

with open(id2name_path, 'r') as f:
    items = f.readlines()

item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
item_ids = [_ for _ in range(len(item_names))]
item_dict = dict()
for i in range(len(item_names)):
    if item_names[i] not in item_dict:
        item_dict[item_names[i]] = [item_ids[i]]
    else:   
        item_dict[item_names[i]].append(item_ids[i])

# 加载嵌入矩阵
def load_embedding(path, name):
    if not os.path.exists(path):
        print(f"Error: {name} embedding not found at {path}")
        return None
    embedding = torch.load(path, weights_only=True).to(device)
    print(f"Loaded {name} embeddings with shape: {embedding.shape}")
    return embedding

# 加载预测嵌入和关键词嵌入
predict_embedding_path = os.path.join(script_dir, 'embed/item_embedding-llama-7b-base.pt')
keywords_embedding_path = os.path.join(script_dir, 'embed/item_embedding-llama-7b-keywords.pt')

movie_embedding_predict = load_embedding(predict_embedding_path, "predict")
movie_embedding_keywords = load_embedding(keywords_embedding_path, "keywords")

if movie_embedding_predict is None or movie_embedding_keywords is None:
    print("Error: Missing embedding files")
    exit(1)

# 准备结果字典
result_dict = dict()

# 批处理函数
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

# 处理每个JSON文件
for p in path:
    if p.endswith("_evaluation.json"):
        continue
    if "llama" in base_model.lower() and "qwen" in p.lower():
        print(f"Skipping Qwen file for Llama model: {p}")
        continue
    if "qwen" in base_model.lower() and "llama" in p.lower():
        print(f"Skipping Llama file for Qwen model: {p}")
        continue
        
    print(f"\nProcessing file: {p}")
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    
    # 设置模型token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    
    # 加载测试数据
    with open(p, 'r') as f:
        test_data = json.load(f)
    
    filtered_test_data = []
    predict_text = []  # 存储predict文本
    keywords_text = []  # 存储keywords文本
    
    # 过滤数据并提取文本
    for data in test_data:
        # 处理predict字段
        predict_cleaned = ""
        if "predict" in data and data["predict"] is not None:
            if isinstance(data["predict"], str):
                predict_cleaned = data["predict"].strip().strip('"')
            elif isinstance(data["predict"], list) and len(data["predict"]) > 0:
                first_pred = data["predict"][0]
                if isinstance(first_pred, str):
                    predict_cleaned = first_pred.strip().strip('"')
        
        # 处理keywords字段
        keywords_cleaned = ""
        if "keywords" in data and data["keywords"] is not None:
            keywords_cleaned = data["keywords"].strip().strip('"')
        
        # 确保两个字段都有有效内容
        if predict_cleaned and keywords_cleaned:
            filtered_test_data.append(data)
            predict_text.append(predict_cleaned)
            keywords_text.append(keywords_cleaned)
    
    test_data = filtered_test_data
    if not test_data:
        print(f"No valid data found in {p}")
        continue
        
    print(f"Processing {len(test_data)} valid samples")
    tokenizer.padding_side = "left"
    
    # 生成predict嵌入
    predict_embeddings = []
    from tqdm import tqdm
    for batch_input in tqdm(batch(predict_text, 4), total=len(predict_text)//4+1, desc="Generating predict embeddings"):
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
    
    # 生成keywords嵌入
    keywords_embeddings = []
    for batch_input in tqdm(batch(keywords_text, 4), total=len(keywords_text)//4+1, desc="Generating keywords embeddings"):
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
        keywords_embeddings.append(hidden_states[-1][:, -1, :].detach())
    
    keywords_embeddings = torch.cat(keywords_embeddings, dim=0)
    
    # 计算距离
    dist_predict = torch.cdist(predict_embeddings, movie_embedding_predict, p=2)
    dist_keywords = torch.cdist(keywords_embeddings, movie_embedding_keywords, p=2)
    
    # 计算排名
    rank_predict = dist_predict.argsort(dim=-1).argsort(dim=-1)
    rank_keywords = dist_keywords.argsort(dim=-1).argsort(dim=-1)
    
    # 加权合并排名
    alpha = args.predict_weight
    beta = args.keywords_weight
    combined_rank = (alpha * rank_predict.float() + beta * rank_keywords.float()).argsort(dim=-1).argsort(dim=-1)
    
    # 计算指标
    topk_list = [1, 3, 5, 10, 20, 50]
    NDCG = []
    HR = []
    
    for topk in topk_list:
        S = 0
        SS = 0
        LL = len(test_data)
        
        for i in range(len(test_data)):
            target_item = test_data[i]['output'].strip("\"").strip(" ")
            min_rank = float('inf')
            
            if target_item in item_dict:
                for item_id in item_dict[target_item]:
                    if item_id < combined_rank.shape[1]:
                        if combined_rank[i][item_id].item() < min_rank:
                            min_rank = combined_rank[i][item_id].item()
            
            if min_rank < topk:
                S += (1 / math.log(min_rank + 2))  # +2 因为排名从0开始，位置=排名+1
                SS += 1
        
        if LL > 0:  # 避免除以零
            NDCG.append(S / LL / (1.0 / math.log(2)))
            HR.append(SS / LL)
        else:
            NDCG.append(0.0)
            HR.append(0.0)
    
    print(f"File: {p}")
    print(f"Predict weight: {alpha}, Keywords weight: {beta}")
    print(f"NDCG@k: {NDCG}")
    print(f"HR@k: {HR}")
    print('_' * 100)
    
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR

# 保存结果
output_dir = os.path.join(script_dir, 'data/game/result/base')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'game-combined_evaluation.json')

with open(output_path, 'w') as f:    
    json.dump(result_dict, f, indent=4)

print(f"Evaluation results saved to: {output_path}")