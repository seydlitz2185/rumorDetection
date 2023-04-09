#!/bin/zsh

# Run all the scripts in the current directory
cd ~/Chinese-Text-Classification-Pytorch;
rm ./pygp_data/saved_dict/*.ckpt;
python run.py --model FastText --embedding embedding_cc.zh.300.npz  --word Ture;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture ;
python run.py --model FastText_raw --embedding embedding_cc.zh.300.npz  --word True;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture;

python run.py --model Transformer --embedding embedding_cc.zh.300.npz  --word True;
python result_to_csv.py;