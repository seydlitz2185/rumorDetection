#!/bin/zsh

# Run all the scripts in the current directory
rm ./pygp_data/saved_dict/*.ckpt;
python run.py --model FastText  --word Ture ;
python run.py --model FastText_gp_raw  --word True;
python run.py --model FastText_gp_sentiment  --word Ture ;
python run.py --model FastText_gp_pca_100  --word Ture ;
python run.py --model FastText_gp_pca_100_sentiment  --word Ture ;
python run.py --model FastText_gp_pca_300  --word Ture ;
python run.py --model FastText_gp_pca_300_sentiment  --word Ture ;
python run.py --model FastText_gp_ldia_100  --word Ture ;
python run.py --model FastText_gp_ldia_100_sentiment  --word Ture ;
python run.py --model FastText_gp_ldia_300  --word Ture ;
python run.py --model FastText_gp_ldia_300_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_pca_100  --word Ture ;
python run.py --model FastText_gp_raw_pca_100_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_pca_300  --word Ture ;
python run.py --model FastText_gp_raw_pca_300_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_ldia_100  --word Ture ;
python run.py --model FastText_gp_raw_ldia_100_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_ldia_300  --word Ture ;
python run.py --model FastText_gp_raw_ldia_300_sentiment  --word Ture ;
python run.py --model Transformer --word True;
python result_to_csv.py;