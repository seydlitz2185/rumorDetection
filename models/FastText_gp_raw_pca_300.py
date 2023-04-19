# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        embedding = 'embedding_cc.zh.300.npz'
        self.model_name = 'FastText_gp_raw_pca_300'                         # 模型名称  
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                      # 预训练词向量
        self.embedding_pretrained_topics = torch.tensor(
            np.load(dataset + '/data/embedding_pca.zh.300.npz' )["embeddings"].astype('float32'))                              # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')   # 设备
        self.topics = self.embedding_pretrained_topics.shape[1]                                                          # pca主题数 
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 30                                         # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 100                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = (self.topics)*2                                        # 隐藏层大小
        self.n_gram_vocab = 150000                                    # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_topics = nn.Embedding.from_pretrained(config.embedding_pretrained_topics, freeze=False)
        self.dropout = nn.Dropout(config.dropout)
        #全联接层1
        self.fc1 = nn.Linear(config.topics, config.hidden_size)
        # 全连接层2，输出层
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        if(len(x) == 5):
            out_topics = self.embedding_topics(x[4])
            out = torch.cat((out_topics,), -1)
        out = out.mean(dim=1)
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
