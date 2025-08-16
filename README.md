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
|      Model          |Sample|Epoch|Method|NG@1|NG@3|NG@5|NG@10|NG@20|NG@50|HR@1|HR@3|HR@5|HR@10|HR@20|HR@50|
|---------------------|------|----|--------|----|----|----|----|----|----|----|----|----|----|----|----|
|      llama-7b       |    0 | 0|  None  + basic  |0.0004|0.0009|0.0011|0.0013|0.0018|0.0026|0.0004|0.0012|0.0018|0.0024|0.0042|0.0086|
|      llama-7b       |    0 | 0|  None  +Enhanced|0.0086|0.0099|0.0105|0.0114|0.0124|0.0146|0.0086|0.0108|0.0122|0.0150|0.0193|0.0301|
|      llama-7b       | 1024 | 1|Enhanced+Enhanced|0.0030|0.0057|0.0068|0.0087|0.0105|0.0130|0.0030|0.0078|0.0104|0.0162|0.0236|0.0360|
|      llama-7b       | 1024 | 2|Enhanced+Enhanced|0.0048|0.0084|0.0093|0.0123|0.0146|0.0184|0.0048|0.0114|0.0134|0.0226|0.0320|0.0510|
|      llama-7b       | 1024 | 3| Basic  + basic  |0.0102|0.0131|0.0144|0.0174|0.0201|0.0242|0.0102|0.0154|0.0186|0.0278|0.0386|0.0590|
|      llama-7b       | 1024 | 3|Enhanced+Enhanced|0.0110|0.0142|0.0156|0.0184|0.0204|0.0251|0.0110|0.0168|0.0200|0.0288|0.0366|0.0602|
|      llama-7b       | 1024 | 4| Basic  + basic  |0.0116|0.0149|0.0161|0.0188|0.0217|0.0258|0.0116|0.0176|0.0204|0.0290|0.0404|0.0612|
|      llama-7b       | 1024 | 4|Enhanced+Enhanced|0.0126|0.0162|0.0178|0.0209|0.0231|0.0275|0.0126|0.0190|0.0228|0.0322|0.0410|0.0638|
|      llama-7b       | 1024 | 5| Basic  + basic  |0.0110|0.0149|0.0163|0.0194|0.0222|0.0269|0.0110|0.0178|0.0212|0.0310|0.0042|0.0658|
|      llama-7b       | 1024 | 5|Enhanced+Enhanced|0.0128|0.0168|0.0187|0.0222|0.0248|0.0291|0.0128|0.0196|0.0242|0.0352|0.0456|0.0672|
|      llama-7b       | 1024 | 6|Enhanced+Enhanced|0.0139|0.0181|0.0207|0.0235|0.0264|0.0308|0.0139|0.0211|0.0275|0.0362|0.0480|0.0700|
|      llama-7b       | 1024 | 7|Enhanced+Enhanced|0.0153|0.0197|0.0223|0.0250|0.0282|0.0331|0.0153|0.0228|0.0292|0.0372|0.0502|0.0747|
|      llama-7b       | 1024 | 8|Enhanced+Enhanced|0.0148|0.0201|0.0221|0.0247|0.0278|0.0318|0.0148|0.0242|0.0290|0.0371|0.0493|0.0694|
|      llama-7b       | 1024 | 9|Enhanced+Enhanced|0.0147|0.0186|0.0221|0.0252|0.0281|0.0329|0.0147|0.0215|0.0300|0.0398|0.0511|0.0747|
|      llama-7b       | 1024 |10|Enhanced+Enhanced|0.0149|0.0195|0.0216|0.0239|0.0274|0.0325|0.0149|0.0227|0.0279|0.0351|0.0489|0.0747|
| llama-7b-instruct   |    0 | 0|  None  + basic  |0.0008|0.0014|0.0016|0.0019|0.0022|0.0032|0.0008|0.0018|0.0024|0.0032|0.0046|0.0094|
| llama-7b-instruct   | 1024 | 0|  None  +Enhanced|0.0044|0.0058|0.0066|0.0075|0.0093|0.0117|0.0044|0.0068|0.0086|0.0116|0.0188|0.0308|
| llama-7b-instruct   | 1024 | 1|  None  +Enhanced|0.0062|0.0081|0.0093|0.0117|0.0144|0.0185|0.0062|0.0094|0.0124|0.0198|0.0306|0.0512|
|*llama-7b-instruct*  | 1024 | 2|Enhanced+Enhanced|0.0106|0.0146|0.0165|0.0194|0.0226|0.0277|0.0106|0.0176|0.0222|0.0312|0.0440|0.0694|
| llama-7b-instruct   | 1024 | 3|Enhanced+ basic  |0.0120|0.0156|0.0181|0.0203|0.0237|0.0292|0.0120|0.0184|0.0244|0.0314|0.0448|0.0724|
|*llama-7b-instruct*  | 1024 | 3|Enhanced+Enhanced|0.0130|0.0160|0.0174|0.0203|0.0237|0.0282|0.0130|0.0184|0.0216|0.0306|0.0440|0.0668|
| llama-7b-instruct   | 1024 | 4|Enhanced+ basic  |0.0124|0.0161|0.0180|0.0205|0.0237|0.0288|0.0124|0.0190|0.0234|0.0312|0.0440|0.0698|
| llama-7b-instruct   | 1024 | 4|Enhanced+Enhanced|0.0110|0.0146|0.0162|0.0187|0.0221|0.0270|0.0110|0.0174|0.0212|0.0290|0.0428|0.0676|
| llama-7b-instruct   | 1024 | 5|Enhanced+ basic  |0.0120|0.0167|0.0192|0.0217|0.0251|0.0300|0.0120|0.0202|0.0264|0.0340|0.0476|0.0722|
| llama-7b-instruct   | 1024 | 5|Enhanced+Enhanced|0.0112|0.0149|0.0167|0.0192|0.0225|0.0275|0.0112|0.0176|0.0220|0.0298|0.0430|0.0682|
| llama-7b-instruct   | 1024 | 6|Enhanced+Enhanced|0.0126|0.0164|0.0190|0.0217|0.0251|0.0298|0.0126|0.0194|0.0256|0.0340|0.0476|0.0714|
| llama-7b-instruct   | 1024 | 7|Enhanced+Enhanced|0.0134|0.0174|0.0198|0.0229|0.0264|0.0315|0.0134|0.0206|0.0264|0.0360|0.0498|0.0758|
| llama-7b-instruct   | 1024 | 8|Enhanced+Enhanced|0.0132|0.0173|0.0191|0.0215|0.0251|0.0300|0.0132|0.0204|0.0248|0.0322|0.0466|0.0716|
| llama-7b-instruct   | 1024 | 9|Enhanced+Enhanced|0.0126|0.0165|0.0182|0.0211|0.0242|0.0292|0.0126|0.0194|0.0236|0.0326|0.0450|0.0702|
| llama-7b-instruct   | 1024 |10|Enhanced+Enhanced|0.0122|0.0162|0.0185|0.0214|0.0247|0.0297|0.0122|0.0194|0.0250|0.0338|0.0472|0.0724|
|Qwen2.5-0.5B-Instruct|    0 | 0|  None  +Enhanced|0.0082|0.0103|0.0116|0.0137|0.0165|0.0191|0.0082|0.0120|0.0152|0.0214|0.0326|0.0462|
|Qwen2.5-0.5B-Instruct| 1024 | 1|Enhanced+Enhanced|0.0084|0.0108|0.0118|0.0139|0.0166|0.0195|0.0084|0.0128|0.0152|0.0218|0.0324|0.0474|
|Qwen2.5-0.5B-Instruct| 1024 | 2|Enhanced+Enhanced|0.0088|0.0112|0.0121|0.0139|0.0167|0.0199|0.0088|0.0130|0.0152|0.0212|0.0322|0.0480|
|Qwen2.5-0.5B-Instruct| 1024 | 3|Enhanced+Enhanced|0.0100|0.0124|0.0132|0.0149|0.0181|0.0210|0.0100|0.0142|0.0162|0.0214|0.0342|0.0490|
|Qwen2.5-0.5B-Instruct| 1024 | 4|Enhanced+Enhanced|0.0082|0.0102|0.0116|0.0136|0.0159|0.0183|0.0082|0.0118|0.0152|0.0214|0.0308|0.0430|
|Qwen2.5-0.5B-Instruct| 1024 | 5|Enhanced+Enhanced|0.0084|0.0100|0.0110|0.0123|0.0143|0.0167|0.0084|0.0112|0.0136|0.0178|0.0262|0.0376|
|Qwen2.5-0.5B-Instruct| 1024 | 6|Enhanced+Enhanced|0.0080|0.0096|0.0107|0.0125|0.0145|0.0167|0.0080|0.0108|0.0132|0.0192|0.0272|0.0384|
|Qwen2.5-0.5B-Instruct| 1024 | 7|Enhanced+Enhanced|0.0092|0.0136|0.0126|0.0142|0.0162|0.0185|0.0092|0.0130|0.0160|0.0210|0.0292|0.0406|
|Qwen2.5-0.5B-Instruct| 1024 | 8|Enhanced+Enhanced|0.0090|0.0105|0.0118|0.0135|0.0158|0.0184|0.0090|0.0116|0.0148|0.0200|0.0292|0.0424|
|Qwen2.5-0.5B-Instruct| 1024 | 9|Enhanced+Enhanced|0.0090|0.0103|0.0117|0.0137|0.0160|0.0186|0.0090|0.0112|0.0146|0.0210|0.0302|0.0432|
|Qwen2.5-0.5B-Instruct| 1024 |10|Enhanced+Enhanced|0.0086|0.0104|0.0116|0.0134|0.0153|0.0185|0.0086|0.0118|0.0146|0.0204|0.0282|0.0440|
| Qwen2.5-3B-Instruct |    0 | 0|  None  +Enhanced|0.0028|0.0042|0.0055|0.0073|0.0100|0.0137|0.0028|0.0052|0.0084|0.0140|0.0248|0.0436|
| Qwen2.5-3B-Instruct | 1024 | 1|Enhanced+Enhanced|0.0030|0.0047|0.0056|0.0074|0.0101|0.0138|0.0030|0.0060|0.0082|0.0136|0.0242|0.0434|
| Qwen2.5-3B-Instruct | 1024 | 2|Enhanced+Enhanced|0.0032|0.0054|0.0065|0.0082|0.0109|0.0155|0.0032|0.0070|0.0096|0.0148|0.0256|0.0490|
| Qwen2.5-3B-Instruct | 1024 | 3|Enhanced+Enhanced|0.0034|0.0060|0.0080|0.0103|0.0129|0.0179|0.0034|0.0080|0.0128|0.0196|0.0304|0.0554|
|*Qwen2.5-3B-Instruct*| 1024 | 4|Enhanced+Enhanced|0.0116|0.0146|0.0167|0.0189|0.0214|0.0260|0.0116|0.0168|0.0220|0.0288|0.0388|0.0620|
| Qwen2.5-3B-Instruct | 1024 | 5|Enhanced+Enhanced|0.0106|0.0141|0.0160|0.0182|0.0216|0.0253|0.0106|0.0168|0.0214|0.0280|0.0416|0.0608|
| Qwen2.5-3B-Instruct | 1024 | 6|Enhanced+Enhanced|0.0102|0.0132|0.0151|0.0172|0.0201|0.0234|0.0102|0.0156|0.0200|0.0266|0.0384|0.0554|
| Qwen2.5-3B-Instruct | 1024 | 7|Enhanced+Enhanced|0.0112|0.0148|0.0165|0.0184|0.0212|0.0250|0.0112|0.0176|0.0218|0.0274|0.0384|0.0580|
| Qwen2.5-3B-Instruct | 1024 | 8|Enhanced+Enhanced|0.0110|0.0150|0.0167|0.0188|0.0220|0.0257|0.0110|0.0182|0.0224|0.0288|0.0414|0.0604|
| Qwen2.5-3B-Instruct | 1024 | 9|Enhanced+Enhanced|0.0104|0.0142|0.0160|0.0180|0.0205|0.0240|0.0104|0.0172|0.0216|0.0276|0.0376|0.0556|
| Qwen2.5-3B-Instruct | 1024 |10|Enhanced+Enhanced|0.0102|0.0138|0.0158|0.0178|0.0209|0.0249|0.0102|0.0166|0.0214|0.0278|0.0398|0.0604|
| Qwen2.5-7B-Instruct | 1024 | 0|  None  +Enhanced|0.0002|0.0006|0.0009|0.0014|0.0020|0.0031|0.0002|0.0010|0.0016|0.0034|0.0058|0.0114|
| Qwen2.5-7B-Instruct | 1024 | 1|Enhanced+Enhanced|0.0012|0.0022|0.0027|0.0039|0.0052|0.0067|0.0012|0.0030|0.0042|0.0080|0.0132|0.0212|
| Qwen2.5-7B-Instruct | 1024 | 2|Enhanced+Enhanced|0.0124|0.0150|0.0162|0.0181|0.0210|0.0257|0.0124|0.0168|0.0198|0.0258|0.0372|0.0612|
| Qwen2.5-7B-Instruct | 1024 | 3|Enhanced+Enhanced|0.0120|0.0148|0.0168|0.0192|0.0223|0.0266|0.0120|0.0172|0.0218|0.0294|0.0416|0.0636|
|*Qwen2.5-7B-Instruct*| 1024 | 4|Enhanced+Enhanced|0.0124|0.0161|0.0179|0.0205|0.0234|0.0276|0.0124|0.0188|0.0232|0.0312|0.0426|0.0636|
| Qwen2.5-7B-Instruct | 1024 | 5|Enhanced+Enhanced|0.0110|0.0138|0.0156|0.0178|0.0202|0.0243|0.0110|0.0162|0.0204|0.0274|0.0366|0.0576|
| Qwen2.5-7B-Instruct | 1024 | 6|Enhanced+Enhanced|0.0100|0.0133|0.0150|0.0169|0.0196|0.0235|0.0100|0.0158|0.0198|0.0258|0.0364|0.0560|
| Qwen2.5-7B-Instruct | 1024 | 7|Enhanced+Enhanced|0.0110|0.0141|0.0164|0.0185|0.0212|0.0258|0.0110|0.0166|0.0220|0.0286|0.0392|0.0628|
| Qwen2.5-7B-Instruct | 1024 | 8|Enhanced+Enhanced|0.0106|0.0136|0.0158|0.0182|0.0212|0.0256|0.0106|0.0158|0.0212|0.0288|0.0404|0.0630|
| Qwen2.5-7B-Instruct | 1024 | 9|Enhanced+Enhanced|0.0106|0.0147|0.0165|0.0190|0.0221|0.0266|0.0106|0.0178|0.0220|0.0298|0.0420|0.0654|
| Qwen2.5-7B-Instruct | 1024 |10|Enhanced+Enhanced|0.0102|0.0133|0.0154|0.0180|0.0213|0.0258|0.0102|0.0158|0.0208|0.0288|0.0420|0.0646|
| Qwen2.5-3B-Instruct |    0 | 0|  None  + GRPO   |0.0042|0.0056|0.0070|0.0086|0.0111|0.0156|0.0042|0.0066|0.0100|0.0146|0.0248|0.0476|
| Qwen2.5-3B-Instruct | 1024 | 4|   SFT  + GRPO   |0.0118|0.0151|0.0170|0.0188|0.0220|0.0265|0.0118|0.0176|0.0222|0.0280|0.0406|0.0638|
|Qwen2.5-1.5B-Instruct| 1024 | 0|  None  +Enhanced|0.0016|0.0018|0.0021|0.0029|0.0038|0.0050|0.0016|0.0020|0.0026|0.0052|0.0086|0.0148|
|Qwen2.5-1.5B-Instruct| 1024 | 1|  None  +Enhanced|0.0022|0.0036|0.0043|0.0057|0.0068|0.0092|0.0022|0.0046|0.0064|0.0106|0.0152|0.0274|
|Qwen2.5-1.5B-Instruct| 1024 | 2|Enhanced+Enhanced|0.0100|0.0127|0.0141|0.0166|0.0195|0.0235|0.0100|0.0148|0.0182|0.0256|0.0374|0.0574|
|Qwen2.5-1.5B-Instruct| 1024 | 3|Enhanced+Enhanced|0.0102|0.0131|0.0146|0.0174|0.0207|0.0252|0.0102|0.0154|0.0190|0.0276|0.0410|0.0640|
|Qwen2.5-1.5B-Instruct| 1024 | 4|Enhanced+Enhanced|0.0106|0.0141|0.0158|0.0186|0.0216|0.0261|0.0106|0.0168|0.0208|0.0296|0.0418|0.0644|
|Qwen2.5-1.5B-Instruct| 1024 | 5|Enhanced+Enhanced|0.0104|0.0135|0.0151|0.0175|0.0206|0.0248|0.0104|0.0158|0.0198|0.0274|0.0396|0.0610|
|Qwen2.5-1.5B-Instruct| 1024 | 6|Enhanced+Enhanced|0.0104|0.0146|0.0160|0.0182|0.0218|0.0260|0.0104|0.0178|0.0212|0.0280|0.0422|0.0638|
|Qwen2.5-1.5B-Instruct| 1024 | 7|Enhanced+Enhanced|0.0094|0.0137|0.0157|0.0181|0.0212|0.0257|0.0094|0.0170|0.0218|0.0294|0.0418|0.0640|
|Qwen2.5-1.5B-Instruct| 1024 | 8|Enhanced+Enhanced|0.0096|0.0137|0.0153|0.0179|0.0211|0.0257|0.0096|0.0168|0.0206|0.0286|0.0414|0.0650|
|Qwen2.5-1.5B-Instruct| 1024 | 9|Enhanced+Enhanced|0.0098|0.0140|0.0159|0.0185|0.0216|0.0256|0.0098|0.0172|0.0218|0.0298|0.0424|0.0626|
|Qwen2.5-1.5B-Instruct| 1024 |10|Enhanced+Enhanced|0.0108|0.0147|0.0165|0.0188|0.0217|0.0257|0.0108|0.0176|0.0220|0.0292|0.0408|0.0610|
|Qwen2.5-1.5B-Instruct| 10000| 1|Enhanced+Enhanced|0.0106|0.0148|0.0165|0.0188|0.0221|0.0265|0.0106|0.0178|0.0220|0.0292|0.0428|0.0646|
|Qwen2.5-1.5B-Instruct| 10000| 2|Enhanced+Enhanced|0.0116|0.0157|0.0175|0.0201|0.0234|0.0282|0.0116|0.0186|0.0230|0.0312|0.0442|0.0688|
|Qwen2.5-1.5B-Instruct| 10000| 3|Enhanced+Enhanced|0.0106|0.0148|0.0165|0.0188|0.0222|0.0265|0.0106|0.0178|0.0220|0.0292|0.0428|0.0646|
|Qwen2.5-1.5B-Instruct| 10000| 4|Enhanced+Enhanced|0.0110|0.0148|0.0164|0.0187|0.0226|0.0265|0.0110|0.0176|0.0214|0.0284|0.0444|0.0636|
|Qwen2.5-1.5B-Instruct| 10000| 5|Enhanced+Enhanced|0.0108|0.0145|0.0162|0.0184|0.0219|0.0264|0.0108|0.0172|0.0214|0.0282|0.0420|0.0648|




beta=0.01
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0086|0.0114|0.0129|0.0151|0.0177|0.0206|0.0086|0.0136|0.0174|0.0240|0.0344|0.0494|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0084|0.0112|0.0124|0.0144|0.0171|0.0202|0.0084|0.0134|0.0162|0.0226|0.0334|0.0490|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0086|0.0114|0.0132|0.0154|0.0184|0.0213|0.0086|0.0136|0.0180|0.0248|0.0368|0.0518|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0084|0.0110|0.0125|0.0145|0.0172|0.0202|0.0084|0.0130|0.0168|0.0228|0.0340|0.0490|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0084|0.0116|0.0130|0.0150|0.0180|0.0208|0.0084|0.0142|0.0176|0.0236|0.0358|0.0500|

beta=0.01 only similarity
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0080|0.0112|0.0129|0.0153|0.0175|0.0202|0.0080|0.0138|0.0178|0.0252|0.0342|0.0482|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0084|0.0113|0.0129|0.0153|0.0177|0.0206|0.0084|0.0136|0.0176|0.0250|0.0346|0.0494|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0084|0.0109|0.0124|0.0148|0.0172|0.0200|0.0084|0.0128|0.0166|0.0238|0.0338|0.0478|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0082|0.0107|0.0123|0.0149|0.0174|0.0206|0.0082|0.0128|0.0166|0.0246|0.0346|0.0510|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0084|0.0110|0.0123|0.0143|0.0168|0.0196|0.0084|0.0130|0.0162|0.0224|0.0326|0.0466|

beta=0.1 
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0084|0.0113|0.0129|0.0151|0.0176|0.0206|0.0084|0.0136|0.0174|0.0242|0.0344|0.0498|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0084|0.0114|0.0130|0.0153|0.0178|0.0210|0.0084|0.0138|0.0176|0.0248|0.0346|0.0512|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0086|0.0112|0.0126|0.0148|0.0175|0.0206|0.0086|0.0132|0.0168|0.0236|0.0344|0.0498|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0080|0.0110|0.0126|0.0147|0.0174|0.0204|0.0080|0.0134|0.0172|0.0238|0.0346|0.0494|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0076|0.0100|0.0114|0.0136|0.0165|0.0194|0.0076|0.0120|0.0154|0.0220|0.0336|0.0486|

beta=0.1 only similarity
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0082|0.0108|0.0123|0.0143|0.0171|0.0201|0.0082|0.0128|0.0164|0.0226|0.0338|0.0492|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0080|0.0114|0.0126|0.0145|0.0175|0.0205|0.0080|0.0140|0.0170|0.0230|0.0350|0.0504|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0084|0.0112|0.0127|0.0151|0.0177|0.0204|0.0084|0.0134|0.0172|0.0246|0.0350|0.0484|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0080|0.0107|0.0121|0.0140|0.0168|0.0197|0.0080|0.0128|0.0162|0.0220|0.0332|0.0482|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0078|0.0105|0.0120|0.0143|0.0168|0.0198|0.0078|0.0126|0.0162|0.0234|0.0334|0.0486|

beta=0.5
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0082|0.0110|0.0124|0.0142|0.0168|0.0199|0.0082|0.0132|0.0166|0.0222|0.0326|0.0484|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0080|0.0111|0.0126|0.0145|0.0176|0.0208|0.0080|0.0136|0.0170|0.0232|0.0354|0.0518|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0080|0.0110|0.0123|0.0143|0.0171|0.0201|0.0080|0.0134|0.0166|0.0226|0.0340|0.0494|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0088|0.0118|0.0135|0.0156|0.0184|0.0215|0.0088|0.0142|0.0184|0.0250|0.0360|0.0518|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0080|0.0107|0.0120|0.0140|0.0168|0.0197|0.0080|0.0128|0.0160|0.0220|0.0334|0.0478|

beta=0.5 only similarity
|Qwen2.5-0.5B-Instruct| 1024 | 1|  GRPO           |0.0082|0.0112|0.0128|0.0152|0.0179|0.0211|0.0082|0.0136|0.0174|0.0250|0.0358|0.0520|
|Qwen2.5-0.5B-Instruct| 1024 | 2|  GRPO           |0.0078|0.0100|0.0113|0.0139|0.0164|0.0196|0.0078|0.0118|0.0150|0.0232|0.0334|0.0492|
|Qwen2.5-0.5B-Instruct| 1024 | 3|  GRPO           |0.0078|0.0105|0.0122|0.0143|0.0167|0.0196|0.0078|0.0126|0.0166|0.0230|0.0328|0.0474|
|Qwen2.5-0.5B-Instruct| 1024 | 4|  GRPO           |0.0082|0.0107|0.0119|0.0140|0.0170|0.0198|0.0082|0.0128|0.0156|0.0220|0.0338|0.0480|
|Qwen2.5-0.5B-Instruct| 1024 | 5|  GRPO           |0.0080|0.0115|0.0128|0.0152|0.0181|0.0210|0.0080|0.0142|0.0174|0.0248|0.0364|0.0512|

Latent
|Qwen2.5-0.5B-Instruct|    0 | 0|  None  +Enhanced|0.0084|0.0114|0.0127|0.0149|0.0178|0.0209|0.0084|0.0140|0.0170|0.0240|0.0356|0.0512|








|      Model     |Trainable parameters|Total parameters|Percentage|
|----------------|--------|----|----|
|    llama-7b    |4194304|3504607232|0.1197%|
|CodeLlama-7b-Instruct-hf|4194304|3504738304|0.1197%|
|Qwen2.5-0.5B-Instruct|540672|315660160|0.1713%|
|Qwen2.5-1.5B-Instruct|1089536|889705984|0.1225%|
|Qwen2.5-3B-Instruct|1843200|1700515840|0.1084%|
|Qwen2.5-7B-Instruct|2523136|4355495424|0.0579%|