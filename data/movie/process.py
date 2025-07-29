import csv
import json
import pandas as pd
import random
import numpy as np
import os

f = open('dataset/ratings.dat', 'r')
data = f.readlines()
f = open('dataset/movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1] for _ in movies]
movie_ids = [_.split('::')[0] for _ in movies]
movie_dict = dict(zip(movie_ids, movie_names))
id_mapping = dict(zip(movie_ids, range(len(movie_ids))))

interaction_dicts = dict()
for line in data:
    user_id, movie_id, rating, timestamp = line.split('::')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
            'movie_title': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(float(rating) > 3.0))
    interaction_dicts[user_id]['timestamp'].append(timestamp)
    interaction_dicts[user_id]['movie_title'].append(movie_dict[movie_id])

os.makedirs('dataset/processed', exist_ok=True)
with open('dataset/processed/all.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_id', 'rating', 'timestamp', 'item_title'])
    for user_id, user_dict in interaction_dicts.items():
        writer.writerow([user_id, user_dict['movie_id'], user_dict['rating'], user_dict['timestamp'], user_dict['movie_title']])

sequential_interaction_list = []
seq_len = 10
for user_id in interaction_dicts:
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'], interaction_dicts[user_id]['movie_title'])
    temp = sorted(temp, key=lambda x: int(x[2]))
    result = zip(*temp)
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'], interaction_dicts[user_id]['movie_title'] = [list(_) for _ in result]
    for i in range(10, len(interaction_dicts[user_id]['movie_id'])):
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_title'][i - seq_len: i],interaction_dicts[user_id]['movie_id'][i-seq_len:i], interaction_dicts[user_id]['rating'][i-seq_len:i], interaction_dicts[user_id]['movie_id'][i], interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )
print(len(sequential_interaction_list))

sequential_interaction_list = sorted(sequential_interaction_list, key=lambda x: int(x[-1]))
with open('dataset/processed/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_title', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list)*0.8)])
with open('dataset/processed/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_title', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.8):int(len(sequential_interaction_list)*0.9)])
with open('dataset/processed/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_title', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.9):])

def csv_to_json(input_path, output_path, sample=False):
    data = pd.read_csv(input_path)
    if sample:
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
        data.to_csv(output_path[:-5] + ".csv", index=False)
    json_list = []
    for index, row in data.iterrows():
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_movie_title'] = eval(row['history_movie_title'])
        L = len(row['history_movie_id'])
        history = "The user has watched the following movies before:"
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_movie_title'][i] + "\""
            else:
                history += ", \"" + row['history_movie_title'][i] + "\""
        target_movie_name = "\"" + movie_dict[str(row['movie_id'])] + "\""
        json_list.append({
            "instruction": "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_movie_name,
        })    
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

csv_to_json('dataset/processed/train.csv', 'dataset/processed/train.json')
csv_to_json('dataset/processed/valid.csv', 'dataset/processed/valid.json')
csv_to_json('dataset/processed/test.csv', 'dataset/processed/test.json')
csv_to_json('dataset/processed/valid.csv', 'dataset/processed/valid_5000.json', sample=True)
csv_to_json('dataset/processed/test.csv', 'dataset/processed/test_5000.json', sample=True)