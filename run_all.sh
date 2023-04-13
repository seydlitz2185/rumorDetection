#!/bin/zsh

# Run all the scripts in the current directory
rm ./pygp_data/saved_dict/*.ckpt;
python run.py --model FastText_gp_100  --word Ture ;
python run.py --model FastText_gp_100_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_100  --word True;
python run.py --model FastText_gp_raw_100_sentiment  --word Ture ;
python run.py --model FastText_gp_300  --word Ture ;
python run.py --model FastText_gp_300_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_300  --word True;
python run.py --model FastText_gp_raw_300_sentiment  --word Ture ;
python run.py --model FastText_gp_svd_100  --word Ture ;
python run.py --model FastText_gp_svd_100_sentiment  --word Ture ;
python run.py --model FastText_gp_svd_300  --word Ture ;
python run.py --model FastText_gp_svd_300_sentiment  --word Ture ;
python run.py --model FastText_gp_pca_100  --word Ture ;
python run.py --model FastText_gp_pca_100_sentiment  --word Ture ;
python run.py --model FastText_gp_pca_300  --word Ture ;
python run.py --model FastText_gp_pca_300_sentiment  --word Ture ;
python run.py --model FastText_gp_ldia_100  --word Ture ;
python run.py --model FastText_gp_ldia_100_sentiment  --word Ture ;
python run.py --model FastText_gp_ldia_300  --word Ture ;
python run.py --model FastText_gp_ldia_300_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_svd_100  --word Ture ;
python run.py --model FastText_gp_raw_svd_100_sentiment  --word Ture ;
python run.py --model FastText_gp_raw_svd_300  --word Ture ;
python run.py --model FastText_gp_raw_svd_300_sentiment  --word Ture ;
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