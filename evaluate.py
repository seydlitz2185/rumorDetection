# -*- coding: utf-8 -*-
import pickle as pkl
from importlib import import_module
from utils_fasttext import build_iterator
import pandas as pd
import torch
from sklearn.metrics import f1_score
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText,FastText_gp TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--mode',default='1',type=str,help='choose a mode: 1,2,3,4,5,6')
args = parser.parse_args()
mode_dict = {'1':'topic','2':'topic*sentiment','3':'topic*length','4':'topic*cos','5':'topic*linguistic','6':'topic*linguistic*sentiment'}

def load_model(dataset, model_name):
    embedding = 'random'
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    tokenizer = lambda x: [y for y in x]
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    if model_name == 'FastText_gp' or model_name == 'FastText_gp_raw':
         if args.mode != '':
            out_name = mode_dict[args.mode]
            out_name = out_name.replace('*','_')
            config.out_name = '_'+out_name
    config.n_vocab = len(vocab)
    config.batch_size = 1
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))

    return config, tokenizer, model, vocab


def load_dataset(indexs,sens, pad_size=32):
    data = pd.read_csv('data.csv',index_col=0)
    contents = []
    UNK, PAD = '<UNK>', '<PAD>'
    mode_num = int(args.mode)
    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    for i in range(len(sens)):
        content = sens[i]
        label = 0
        words_line = []
        token = tokenizer(content)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        # fasttext ngram
        if model_name == 'FastText' or model_name == 'FastText_raw' :
            buckets = config.n_gram_vocab
            bigram = []
            trigram = []
            # ------ngram------
            for i in range(pad_size):
                bigram.append(biGramHash(words_line, i, buckets))
                trigram.append(triGramHash(words_line, i, buckets))
            # -----------------
            contents.append((words_line, int(label), seq_len, bigram, trigram))
        elif model_name == 'FastText_gp' or model_name == 'FastText_gp_raw':
            index = indexs[i]
            buckets = config.n_gram_vocab
            bigram = []
            trigram = []
            sentiment = np.array([data.loc[index,'sentiments']] if data.loc[index,'sentiments'] != 'nan' else [0])
            cos_sim  = np.array((data.loc[index,'cos_sim']) if data.loc[index,'cos_sim'] != 'nan' else [0])
            length = np.array([data.loc[index,'length']] if data.loc[index,'length'] != 'nan' else [0])
            topic = np.array(eval(data.loc[index,'pca_topic_vectors'])if data.loc[index,'pca_topic_vectors'] != 'nan' else [0]*100)*100000
            if mode_num == 1:
                pass
            elif  mode_num == 2:
                topic = topic*sentiment
            elif  mode_num == 3:
                topic = topic*length
            elif mode_num == 4:
                topic = topic*cos_sim
            elif mode_num == 5:
                topic = topic*length*cos_sim
            elif mode_num == 6:
                topic = topic*length*cos_sim*sentiment
            else:
                raise ValueError('mode_num is wrong')
            # ------ngram------
            for i in range(pad_size):
                bigram.append(biGramHash(words_line, i, buckets))
                trigram.append(triGramHash(words_line, i, buckets))
            # -----------------
            contents.append((words_line, int(label), seq_len, bigram, trigram,topic))
        else:
            contents.append((words_line, int(label), seq_len, ))

    return contents


def evaluate(indexs,sens):
    predicts = []
    model.eval()
    test_data = load_dataset(indexs,sens, config.pad_size)
    test_iter = build_iterator(test_data, config)

    with torch.no_grad():
        for sen, (texts, _) in zip(sens, test_iter):
            outputs = model(texts)
            pred = torch.argmax(outputs.data, 1).cpu().numpy()
            print(f"{sen} 预测结果为 {id2target[pred[0]]} .")
            predicts.append(pred[0])

    return predicts


if __name__ == "__main__":
    id2target = {0: "非谣言", 1: "谣言"}
    dataset = './pygp_data'  # 数据集
    model_name = args.model
    config, tokenizer, model, vocab = load_model(dataset,model_name)
    embedding = 'embedding_cc.zh.300.npz'
    
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText' or model_name == 'FastText_raw':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        #embedding = 'random'
    if model_name == 'FastText_gp' or model_name == 'FastText_gp_raw':
        from utils_fasttext_gp import build_dataset, build_iterator, get_time_dif
        #embedding = 'random'
        dataset = './pygp_data'
        if args.mode != '':
            out_name = mode_dict[args.mode]
            out_name = out_name.replace('*','_')
            config.out_name = '_'+out_name
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    indexs,sens, labels = [], [],[]
    with open(dataset+"/data/test.txt", "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip()
            index,sen, label = line.split("\t")
            indexs.append(index)
            sens.append(sen)
            labels.append(int(label))

    preds = evaluate(indexs,sens)
    assert len(preds) == len(labels)
    print(model_name+' : %.4f' % f1_score(preds, labels, average="macro"))  # 0.9624