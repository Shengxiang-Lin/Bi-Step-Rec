import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os
import csv

with open('dataset/meta_Video_Games.json') as f:
    metadata = [json.loads(line) for line in f]
with open('dataset/Video_Games_5.json') as f:
    reviews = [json.loads(line) for line in f]
users = set()
items = set()
for review in tqdm(reviews):
    users.add(review['reviewerID'])
    items.add(review['asin'])
item2id = dict()
count = 0
for item in items:
    item2id[item] = count
    count += 1
print(len(users), len(items), len(reviews), len(reviews) / (len(users) * len(items)))

id_title = {}
id_item = {}
cnt = 0
for meta in tqdm(metadata):
    if len(meta['title']) > 1: # remove the item without title
        id_title[meta['asin']] = meta['title']

users = dict()
for review in tqdm(reviews):
    user = review['reviewerID']
    if 'asin' not in review:
        break
    item = review['asin']
    if item not in id_title:
        continue
    if review['asin'] not in id_item:
        id_item[review['asin']] = cnt
        cnt += 1
    if 'overall' not in review:
        continue
    if 'unixReviewTime' not in review:
        continue
    if user not in users:
        users[user] = {
            'items': [],
            'ratings': [],
            'timestamps': [],
            'reviews': []
        }
    users[user]['items'].append(item)
    users[user]['ratings'].append(review['overall'])
    users[user]['timestamps'].append(review['unixReviewTime'])

user_id = 0
interactions = []
B = []
for key in tqdm(users.keys()):
    items = users[key]['items']
    ratings = users[key]['ratings']
    timestamps = users[key]['timestamps']
    all = list(zip(items, ratings, timestamps))
    res = sorted(all, key=lambda x: int(x[-1]))
    items, ratings, timestamps = zip(*res)
    items, ratings, timestamps = list(items), list(ratings), list(timestamps)
    users[key]['items'] = items
    users[key]['item_ids'] = [item2id[x] for x in items]
    users[key]['item_titles'] = [id_title[x] for x in items]
    users[key]['ratings'] = ratings
    users[key]['timestamps'] = timestamps
    for i in range(min(10, len(items) - 1), len(items)):
        st = max(i - 10, 0)
        interactions.append([key, users[key]['items'][st: i], users[key]['items'][i], users[key]['item_ids'][st: i], users[key]['item_ids'][i], users[key]['item_titles'][st: i], users[key]['item_titles'][i], ratings[st: i], ratings[i], int(timestamps[i])])   
print(len(interactions))

interactions = sorted(interactions, key=lambda x: x[-1])
os.makedirs('dataset/processed', exist_ok=True)
with open('dataset/processed/train.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp'])
    csvwriter.writerows(interactions[:int(len(interactions) * 0.8)])
with open('dataset/processed/valid.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp'])
    csvwriter.writerows(interactions[int(len(interactions) * 0.8):int(len(interactions) * 0.9)])
with open('dataset/processed/test.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp'])
    csvwriter.writerows(interactions[int(len(interactions) * 0.9):])

def csv_to_json(input_path, output_path, sample=False):
    data = pd.read_csv(input_path)
    if sample:
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
        data.to_csv(output_path[:-5] + ".csv", index=False)
    json_list = []
    for index, row in tqdm(data.iterrows()):
        row['history_item_title'] = eval(row['history_item_title'])
        row['history_rating'] = eval(row['history_rating'])
        L = len(row['history_item_title'])
        history = "The user has played the following video games before:"
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ", \"" + row['history_item_title'][i] + "\""
        target_movie = str(row['item_title'])
        target_movie_str = "\"" + target_movie + "\""
        json_list.append({
            "instruction": "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_movie_str,
        })        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

csv_to_json('dataset/processed/train.csv', 'dataset/processed/train.json')
csv_to_json('dataset/processed/valid.csv', 'dataset/processed/valid.json')
csv_to_json('dataset/processed/test.csv', 'dataset/processed/test.json')
csv_to_json('dataset/processed/valid.csv', 'dataset/processed/valid_5000.json', sample=True)
csv_to_json('dataset/processed/test.csv', 'dataset/processed/test_5000.json', sample=True) 