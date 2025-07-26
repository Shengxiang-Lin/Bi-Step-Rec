# BIGRec

## Download the base_models 
```
python download_base_models.py
```

## Game datasets   
You can download the datasets from
![meta_Video_Games.json](https://www.kaggle.com/datasets/khaledsayed111/meta-video-games)
![Video_Games_5.json](https://www.kaggle.com/code/idrimadrid/projet-text-mining-amazon-reviews/output)
![id2name.txt](https://github.com/SAI990323/BIGRec/blob/main/data/game/id2name.txt)

```
cd data/game  
python process.py
python embedding.py
```
### Train
```
python train_game.py     
#train with Deepspeed: python -m accelerate.commands.launch train_game.py 
```
### Inference
```
python inference.py
```
### Evaluate
```
python data/game/evaluate.py --input_dir data/game/result
```    
## Movie datasets   
You can download the datasets from
![movies.dat](https://grouplens.org/datasets/movielens/10m/)
![rating.dat](https://grouplens.org/datasets/movielens/10m/)
```
cd data/movie   
python process.py
python embedding.py
``` 
### Train
```
python train_movie.py     
#train with Deepspeed: python -m accelerate.commands.launch train_movie.py 
```

### Inference     
```
python inference.py   
```
### Evaluate       
```
python data/movie/evaluate.py --input_dir data/movie/result    
```     
## File directory structure    
```
Current working directory/
├─ base_models/  
│  └─ llama-7b/
├─ bitsandbytes/ 
├─ data/
│  ├─ game/
│  │  ├─ dataset/
│  │  └─ ...
│  └─ movie/
│     ├─ dataset/
│     └─ ...
├─ lora-alpaca-game/
├─ lora-alpaca-movie/
├─ movie_result/
├─ README.md
├─ requirements.txt
├─ train_game.py
├─ train_movie.py
└─ inference.py
```