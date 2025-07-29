# BIGRec

### Download the base_models 
```
python download_base_models.py
```

#### Game datasets   
You can download the datasets from
[meta_Video_Games.json](https://www.kaggle.com/datasets/khaledsayed111/meta-video-games)
[Video_Games_5.json](https://www.kaggle.com/code/idrimadrid/projet-text-mining-amazon-reviews/output)
[id2name.txt](https://github.com/SAI990323/BIGRec/blob/main/data/game/id2name.txt)
```
cd data/game 
python process.py
python embedding.py
```   
#### Movie datasets   
You can download the datasets from
[movies.dat](https://grouplens.org/datasets/movielens/10m/)
[rating.dat](https://grouplens.org/datasets/movielens/10m/)
```
cd data/movie   
python process.py
python embedding.py
```  
### Train (Take movie as an example)
```
python train.py --train_data_path '["data/movie/dataset/processed/train.json"]' --val_data_path '["data/movie/dataset/processed/valid_5000.json"]' --val_test_path '["data/movie/dataset/processed/test_5000.json"]' --output_dir ./lora-alpaca-movie
#train with Deepspeed
python -m accelerate.commands.launch train.py --train_data_path '["data/movie/dataset/processed/train.json"]' --val_data_path '["data/movie/dataset/processed/valid_5000.json"]' --val_test_path '["data/movie/dataset/processed/test_5000.json"]' --output_dir ./lora-alpaca-movie
```

### Inference (Take movie as an example)     
```
python inference.py --lora_weights lora-alpaca-movie/checkpoint-40 --test_data_path data/movie/dataset/processed/test_5000.json result_json_data data/movie/result/movie.json
#train with Deepspeed
python -m accelerate.commands.launch inference.py --lora_weights lora-alpaca-movie/checkpoint-40 --test_data_path data/movie/dataset/processed/test_5000.json result_json_data data/movie/result/movie.json
```

### Train and inference together (Take movie as an example)     
```
python train-inference.py --train_data_path '["data/movie/dataset/processed/train.json"]' --val_data_path '["data/movie/dataset/processed/valid_5000.json"]' --val_test_path '["data/movie/dataset/processed/test_5000.json"]' --output_dir ./lora-alpaca-movie
```

### Evaluate (Take movie as an example)       
```
python data/movie/evaluate.py --input_dir data/movie/result    
```     
### File directory structure    
```
Current working directory/
├─ base_models/  
│  └─ llama-7b/
├─ bitsandbytes/ 
├─ data/
│  ├─ game/
│  │  ├─ dataset/
|  │  │  ├─ meta_Video_Games.json
|  │  │  ├─ Video_Games_5.json
|  │  │  ├─ id2name.txt
|  │  |  ├─ processed/
|  │  |  └─ embedding/ 
│  │  └─ ...
│  └─ movie/
│     ├─ dataset/
│     │  ├─ movies.dat
│     │  ├─ rating.dat
│     │  ├─ processed/
|     │  └─ embedding/  
│     └─ ...
├─ lora-alpaca-game/
├─ lora-alpaca-movie/
├─ README.md
├─ requirements.txt
├─ train.py
├─ train-inference.py
└─ inference.py
```