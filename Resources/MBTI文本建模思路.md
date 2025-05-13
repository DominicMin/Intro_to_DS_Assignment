---
mindmap-plugin: basic
source: "[[MBTI人格分析研究方案 ]]"
---

# MBTI文本建模思路

## 第一阶段：数据清洗&文本基本特征采集
- -  操作顺序
    - remove_url
    - 先采集一些基本特征，再继续清洗
        - word_count帖子词数
        - sentence_quantity帖子句子数
        - upper_ratio帖子大写字母比率
        - readability可读性：用两个参数衡量
            - Flesch Reading Ease
            - Gunning Fog Index
        - 如果全部清洗完再采集这些数据就没有意义了，比如tokenize之后的文本就没有可读性
    - expand_contraction
        - 展开诸如you're之类的连接词
    - tolower
    - remove_punct
    - remove_whitespace
    - totokens
        - 文本token化
    - remove_stopwords
    - post_lemmatize
        - 词形还原，例如将running变成run

## * **特征工程（用于后续分析/建模）：**
-
    - **基本文本特征：** 帖子长度/词语数量、句子数量、平均词长、大写字母比例 15 等。
    - **可读性指标：** 计算文本的可读性分数 15。
-
    - **词袋模型/TF-IDF：** 将文本转换为词频或TF-IDF向量。
-
    - **词嵌入（Word Embeddings）：** 使用预训练模型（如Word2Vec, GloVe, BERT 15）将词语或文本转换为密集向量表示。
-
    - **语言学特征（LIWC）：** 使用Linguistic Inquiry and Word Count (LIWC) 15 或类似词典，分析文本在心理、情感、认知过程等维度上的特征。
-
    - **情感/情绪特征：** 使用情感词典（如SenticNet, NRC Emotion Lexicon 15）或情感分析模型，量化文本的情感极性或具体情绪。