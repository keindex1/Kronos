git clone https://github.com/keindex1/Kronos.git
cd ./Kronos
pip install -r requirements.txt
modelscope download --model AI-ModelScope/Kronos-base --local_dir ./pretrained/Kronos-base
modelscope download --model AI-ModelScope/Kronos-Tokenizer-base --local_dir ./pretrained/Kronos-Tokenizer-base
python train_sequential.py --config ./configs/hmd_002947_kline_5min.yaml