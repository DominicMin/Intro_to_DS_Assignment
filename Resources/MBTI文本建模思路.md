---
mindmap-plugin: basic
source: "[[MBTI人格分析研究方案 ]]"
---

# MBTI文本建模思路

## -  操作顺序
- remove_url
- 特征采集
    - word_count
    - sentence_quantity
    - upper_ratio
    - readability
- expand_contraction
- tolower
- remove_punct
- remove_whitespace
- totokens
- remove_stopwords
- post_lemmatize

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