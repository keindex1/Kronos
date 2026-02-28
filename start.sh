git clone https://github.com/yjkindex/Kronos.git
cd ./Kronos
pip install -r requirements.txt
python train_sequential.py --config ./configs/hmd_002947_kline_5min.yaml
cp -r ./finetuned /mnt/data/