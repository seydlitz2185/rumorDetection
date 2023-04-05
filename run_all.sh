#!/bin/zsh

# Run all the scripts in the current directory
cd ~/Chinese-Text-Classification-Pytorch;
rm ./pygp_data/saved_dict/*;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 1;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 2;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 3;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 4;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 5;
python run.py --model FastText_gp_raw --embedding embedding_cc.zh.300.npz  --word Ture --mode 6;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 1;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 2;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 3;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 4;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 5;
python run.py --model FastText_gp --embedding embedding_cc.zh.300.npz  --word Ture --mode 6;
python run.py --model FastText --embedding embedding_cc.zh.300.npz  --word Ture;
python run.py --model FastText_raw --embedding embedding_cc.zh.300.npz  --word True;
python run.py --model Transformer --embedding embedding_cc.zh.300.npz  --word True;