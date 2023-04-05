# -*- coding: utf-8 -*-
import pickle as pkl
from importlib import import_module
from utils_fasttext import build_iterator

import torch
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

def load_model(dataset, model_name):
    embedding = 'random'
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    tokenizer = lambda x: [y for y in x]
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)
    config.batch_size = 1
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))

    return config, tokenizer, model, vocab


def load_dataset(sens, pad_size=32):
    contents = []

    UNK, PAD = '<UNK>', '<PAD>'

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    for content in sens:
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
        if model_name == 'FastText':
            buckets = config.n_gram_vocab
            bigram = []
            trigram = []
            # ------ngram------
            for i in range(pad_size):
                bigram.append(biGramHash(words_line, i, buckets))
                trigram.append(triGramHash(words_line, i, buckets))
            # -----------------
            contents.append((words_line, int(label), seq_len, bigram, trigram))
        else:
            contents.append((words_line, int(label), seq_len, ))

    return contents


def eval(sens):
    predicts = []
    model.eval()
    test_data = load_dataset(sens, config.pad_size)
    test_iter = build_iterator(test_data, config)

    with torch.no_grad():
        for sen, (texts, _) in zip(sens, test_iter):
            outputs = model(texts)
            pred = torch.argmax(outputs.data, 1).cpu().numpy()
            print(f"{sen} 预测结果为 {id2target[pred[0]]} .")
            predicts.append(pred[0])

    return predicts


if __name__ == "__main__":
    id2target = {0: "真相", 1: "谣言"}
    dataset = './THUCNews'  # 数据集
    model_name = args.model
    config, tokenizer, model, vocab = load_model(dataset,model_name)
    embedding = 'embedding_SougouNews.npz'
    
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    sens =  []
    sens.append(input("input:"))

    preds = eval(sens)