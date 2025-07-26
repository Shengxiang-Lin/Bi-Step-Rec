# BIGRec

download the base_models 
```
python download_base_models.py
```

take the game datasets for example   
you can download the datasets from
meta_Video_Games.json
https://www.kaggle.com/datasets/khaledsayed111/meta-video-games
Video_Games_5.json
https://www.kaggle.com/code/idrimadrid/projet-text-mining-amazon-reviews/output

```
cd data/game  
python process.py
python embedding.py
```
```
python train.py
python inference.py
python data/game/evaluate.py --input_dir data/game/result
```


movie    
```
python train.py
python inference.py  --base_model base_models/llama-7b --lora_weights lora-alpaca-movie  --test_data_path data/movie/test_5000.json --result_json_data ./movie_result/movie.json
python ./data/movie/evaluate.py --input_dir ./movie_result
```
game  
