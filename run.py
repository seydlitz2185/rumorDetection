# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
'''
from ray import tune
search_space = {
    'learning_rate': tune.loguniform(1e-5, 1e-2),
    'num_epochs': tune.randint(5, 21),
    'dropout': tune.uniform(0, 0.5),
    'hidden_size': tune.randint(32, 257),
    'num_layers': tune.randint(1,3)
}
'''
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, FastText_raw,FastText_gp,FastText_gp_raw,FastText_gp_raw_topic,TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'pygp_data' 
    embedding = 'embedding_cc.zh.300.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText' or model_name == 'FastText_raw' :
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
    elif model_name.startswith('FastText_gp'):
        from utils_fasttext_gp import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name.startswith('FastText_gp'):
        vocab, train_data, dev_data, test_data = build_dataset(config,args.word)
    else:
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    print("save_path:"+config.save_path)
    train(config, model, train_iter, dev_iter, test_iter)
