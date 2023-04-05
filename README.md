# 基于神经网络的短视频虚假信息检测

## 在[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch.git)的基础上改进的fastText虚假健康信息分类

* 数据集： `data = pd.read_csv('data.csv',index_col=0)`该数据集为自制数据集，从抖音短视频网络社区中收集健康养生类短视频，使用[vosk-api](https://github.com/alphacep/vosk-api.git)提供的[vosk-model-cn-0.22](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip "下载")模型进行自动语音识别，使用[中文标点符号模型](https://github.com/yeyupiaoling/PunctuationModel.gi)对文本标点符号进行恢复，得到短视频内容信息（data.text，data.seg_text为使用jieba分词、哈工大词表去处停用词后得到的文档）。短视频内容使用人工标注方法标注其中的谣言与非谣言（data.rumor，0为非谣言，1为谣言），目前包含2490条健康信息。
* 特征抽取:根据心理学的观点，人对信息的接受具有双过程机制。基于双过程理论的启发式-系统式模型（HSM）将人对信息的接受过程分为启发式过程和系统式过程。根据HSM，本研究从健康信息数据集中抽取了如下特征：
  1. 启发式特征：这一类特征在人对信息的接受过程中起辅助作用。
     * 语言学特征：
       * data.length：计算data.text长度的平均值avg，再分别用每条信息的data.text除以avg。
       * data.cos_sim：使用sklearn计算文档余弦相似度，按行求和后处以每行和的平均值。
     * 情感特征：
       * data.sentiments：使用SnowNLP计算data.text的情感极性得分，分数在（0-1）之间。
  2. 系统式特征：这一类特征在人对信息的接受过程中起主导作用。
     * 主题特征：
       * data.pca_topic_vectors：使用sklearn中TruncatedSVD抽取每个文档的100维主题得分
       * data.ldia_topic_vectors：使用sklearn中LatentDirichletAllocation抽取每个文档的100维主题得分
* 词嵌入：使用[fastText](https://github.com/facebookresearch/fastText)提供的[vectors-crawl/cc.zh.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz)根据data.seg_text中的词构建embedding_cc.zh.300.npz，其中只有一个元组"embeddings"。
* 改进的fasttext模型：所有文档的pad_size默认为100，如修改，请同步截断主题向量data.(pca|lida)?_topic_vectors。改进模型为models/FastText_gp.py，加入一个主题得分嵌入层，与此前word，bigram和trigram求平均。
* 效果：
