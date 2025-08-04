# BIGRec   
### Environment Configuration
```
conda create -n BIGRec python=3.10
conda activate BIGRec
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
export CUDA_VERSION=121
python setup.py install
cd ..
```
```
git clone https://github.com/huggingface/trl
cd trl
git checkout v0.14-release
pip install -e .[dev]
```

#### [Deepspeed Configuration](https://github.com/Shengxiang-Lin/Post-Training-of-LLMs/tree/main/Deepspeed_example)  
It is not necessary here, but Deepspeed can greatly accelerate training.
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
|- download_base_models.py
├─ train.py
├─ train-inference.py
└─ inference.py
```

# Main results

### Movie Dataset
|Model|Sample|NG@1|NG@3|NG@5|NG@10|NG@20|NG@50|HR@1|HR@3|HR@5|HR@10|HR@20|HR@50|
|--------|----|----|----|----|----|----|----|----|----|----|----|----|----|
|llama-7b|1024|0.0094|0.0143|0.0155|0.0178|0.0205|0.0266|0.0094|0.0182|0.0212|0.0288|0.0396|0.0704|

### Game Dataset(seed=0, temperature=0.2, top_p=0.95, top_k=40, num_beams=1, do_sample=True, max_new_tokens=64)
|      Model      |Sample|Epoch|Prompt|NG@1|NG@3|NG@5|NG@10|NG@20|NG@50|HR@1|HR@3|HR@5|HR@10|HR@20|HR@50|
|-----------------|------|----|--------|----|----|----|----|----|----|----|----|----|----|----|----|
|     llama-7b    |    0 | 0|  None  + basic  |0.0004|0.0009|0.0011|0.0013|0.0018|0.0026|0.0004|0.0012|0.0018|0.0024|0.0042|0.0086|
|     llama-7b    |    0 | 0|  None  +Enhanced|0.0086|0.0099|0.0105|0.0114|0.0124|0.0146|0.0086|0.0108|0.0122|0.0150|0.0193|0.0301|
|     llama-7b    | 1024 | 1|Enhanced+Enhanced|0.0030|0.0057|0.0068|0.0087|0.0105|0.0130|0.0030|0.0078|0.0104|0.0162|0.0236|0.0360|
|     llama-7b    | 1024 | 2|Enhanced+Enhanced|0.0048|0.0084|0.0093|0.0123|0.0146|0.0184|0.0048|0.0114|0.0134|0.0226|0.0320|0.0510|
|     llama-7b    | 1024 | 3| Basic  + basic  |0.0102|0.0131|0.0144|0.0174|0.0201|0.0242|0.0102|0.0154|0.0186|0.0278|0.0386|0.0590|
|     llama-7b    | 1024 | 3|Enhanced+Enhanced|0.0110|0.0142|0.0156|0.0184|0.0204|0.0251|0.0110|0.0168|0.0200|0.0288|0.0366|0.0602|
|     llama-7b    | 1024 | 4| Basic  + basic  |0.0116|0.0149|0.0161|0.0188|0.0217|0.0258|0.0116|0.0176|0.0204|0.0290|0.0404|0.0612|
|     llama-7b    | 1024 | 4|Enhanced+Enhanced|0.0126|0.0162|0.0178|0.0209|0.0231|0.0275|0.0126|0.0190|0.0228|0.0322|0.0410|0.0638|
|     llama-7b    | 1024 | 5| Basic  + basic  |0.0110|0.0149|0.0163|0.0194|0.0222|0.0269|0.0110|0.0178|0.0212|0.0310|0.0042|0.0658|
|     llama-7b    | 1024 | 5|Enhanced+Enhanced|0.0128|0.0160|0.0174|0.0203|0.0237|0.0282|0.0130|0.0184|0.0216|0.0306|0.0440|0.0668|
|     llama-7b    | 1024 | 6|Enhanced+Enhanced|0.0120|0.0168|0.0174|0.0203|0.0237|0.0282|0.0130|0.0184|0.0216|0.0306|0.0440|0.0688|
|llama-7b-instruct|    0 | 0|  None  + basic  |0.0008|0.0014|0.0016|0.0019|0.0022|0.0032|0.0008|0.0018|0.0024|0.0032|0.0046|0.0094|
|llama-7b-instruct|    0 | 0|  None  +Enhanced|0.0044|0.0058|0.0066|0.0075|0.0093|0.0117|0.0044|0.0068|0.0086|0.0116|0.0188|0.0308|
|llama-7b-instruct| 1024 | 3|Enhanced+ basic  |0.0120|0.0156|0.0181|0.0203|0.0237|0.0292|0.0120|0.0184|0.0244|0.0314|0.0448|0.0724|
|llama-7b-instruct| 1024 | 3|Enhanced+Enhanced|0.0130|0.0160|0.0174|0.0203|0.0237|0.0282|0.0130|0.0184|0.0216|0.0306|0.0440|0.0668|
|llama-7b-instruct| 1024 | 4|Enhanced+ basic  |0.0124|0.0161|0.0180|0.0205|0.0237|0.0288|0.0124|0.0190|0.0234|0.0312|0.0440|0.0698|
|llama-7b-instruct| 1024 | 4|Enhanced+Enhanced|0.0110|0.0146|0.0162|0.0187|0.0221|0.0270|0.0110|0.0174|0.0212|0.0290|0.0428|0.0676|
|llama-7b-instruct| 1024 | 5|Enhanced+ basic  |0.0120|0.0167|0.0192|0.0217|0.0251|0.0300|0.0120|0.0202|0.0264|0.0340|0.0476|0.0722|
|llama-7b-instruct| 1024 | 5|Enhanced+Enhanced|0.0112|0.0149|0.0167|0.0192|0.0225|0.0275|0.0112|0.0176|0.0220|0.0298|0.0430|0.0682|



|      Model     |Trainable parameters|Total parameters|Percentage|
|----------------|--------|----|----|
|    llama-7b    |4194304|3504607232|0.1197%|
|CodeLlama-7b-Instruct-hf|4194304|3504738304|0.1197%|
|   Qwen2-0.5B   |540672|494545792|0.1093%|