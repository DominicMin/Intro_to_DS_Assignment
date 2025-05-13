### 引言

在自然语言处理（NLP）和机器学习中，我们无法直接处理原始的文本数据。需要将文本转换成计算机能够理解和处理的数值形式，这个过程称为文本向量化或文本表示。本教程将重点介绍两种主流的文本向量化技术：

1. 词袋模型 (Bag-of-Words, BoW) / TF-IDF: 基于词频的经典方法，将文本表示为稀疏向量。
    
2. 词嵌入 (Word Embeddings): 将词语或文本映射到低维、稠密的向量空间，能够捕捉语义信息。
    

我们将使用 Python 中的常用库来实现这两种方法。

### 关于预处理和向量化的顺序

在进行文本向量化之前，通常需要进行一系列的文本预处理步骤。理解这些步骤与向量化方法的先后关系很重要：

1. 对于词袋模型 (BoW) / TF-IDF:
    

- 顺序： 文本清洗 -> 分词 (Tokenization) -> 去停用词 (Stop Word Removal) -> 词形还原 (Lemmatization) / 词干提取 (Stemming) -> BoW/TF-IDF 向量化。
    
- 原因： 这些向量化方法依赖于一个“干净”的词元列表来构建词汇表和计算词频。去除停用词和进行词形还原有助于减少词汇表的大小，并将语义相近的词归一化，提高向量质量。
    

2. 对于词嵌入 (Word Embeddings):
    

- 平均词嵌入 (Average Word Embeddings): 通常也遵循与 BoW/TF-IDF 类似的预处理顺序（分词 -> 去停用词 -> 词形还原），因为你需要得到规范化的词元去查找预训练模型中的向量。
    
- 上下文词嵌入 (Contextual Embeddings, 如 BERT, Sentence Transformers): 这类模型通常需要较少的预处理。
    

- 顺序： 文本清洗 -> 模型特定的分词 (Tokenization) -> 输入模型获取嵌入向量。
    
- 原因： 这些模型被设计用来理解原始文本的上下文信息。**通常不建议去除停用词或进行词形还原/词干提取**，因为这些操作可能会丢失重要的上下文线索。只需要使用模型配套的分词器即可。
    

本教程将在后续步骤中演示适用于 BoW/TF-IDF 和平均词嵌入的预处理流程。

### 准备工作

1. 安装必要的库

如果你还没有安装以下库，请使用 pip 进行安装：
```python
pip install pandas nltk scikit-learn gensim numpy  
# 如果你想尝试基于BERT的句子嵌入，还需要安装：  
# pip install sentence-transformers  
  ```

2. 下载 NLTK 数据包

NLTK 需要一些数据包来进行分词、去除停用词和词形还原等操作。
```python
# 在 Python 脚本或 Jupyter Notebook 中运行  
import nltk  
import ssl  
  
# 尝试解决 SSL 证书问题 (某些环境下需要)  
try:  
    _create_unverified_https_context = ssl._create_unverified_context  
except AttributeError:  
    pass  
else:  
    ssl._create_default_https_context = _create_unverified_https_context  
  
# 下载所需数据包  
nltk_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']  
for package in nltk_packages:  
    try:  
        # 调整查找路径的逻辑  
        if package == 'punkt':  
            nltk.data.find(f'tokenizers/{package}')  
        elif package in ['stopwords', 'wordnet']:  
            nltk.data.find(f'corpora/{package}')  
        elif package == 'averaged_perceptron_tagger':  
            nltk.data.find(f'taggers/{package}')  
        else:  
            nltk.data.find(package) # 尝试通用查找  
        print(f"NLTK 包 '{package}' 已存在。")  
    except nltk.downloader.DownloadError:  
        print(f"正在下载 NLTK 包 '{package}'...")  
        nltk.download(package)  
        print(f"下载完成。")  
    except LookupError: # 处理通用查找失败  
        print(f"查找 NLTK 包 '{package}' 失败，尝试下载...")  
        try:  
            nltk.download(package)  
            print(f"下载完成。")  
        except Exception as e_download:  
              print(f"下载 NLTK 包 '{package}' 时出错: {e_download}")  
  ```
  

3. 准备示例文本数据

我们假设有一个包含文本数据的 Pandas DataFrame。
```python
import pandas as pd  
  
# 创建示例数据  
data = {  
    'id': [1, 2, 3, 4],  
    'text': [  
        "The quick brown fox jumps over the lazy dog.",  
        "Natural language processing is fascinating.",  
        "Word embeddings capture semantic meaning.",  
        "TF-IDF and BoW are classic text representations."  
    ]  
}  
df = pd.DataFrame(data)  
text_column = 'text' # 指定文本列的名称  
  
print("示例文本数据:")  
print(df)  
  ```

### 文本预处理 (适用于 BoW/TF-IDF 和 平均词嵌入)

无论是词袋模型/TF-IDF还是词嵌入，通常都需要对文本进行预处理，以获得更好的向量表示。
```python
import re  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer  
  
# 获取英文停用词  
stop_words = set(stopwords.words('english'))  
# 初始化词形还原器  
lemmatizer = WordNetLemmatizer()  
  
def preprocess_text(text):  
    """  
    对文本进行预处理：转小写、去除非字母字符、分词、去停用词、词形还原。  
  
    Args:  
        text (str): 输入的原始文本字符串。  
  
    Returns:  
        list: 处理后的词元列表。  
    """  
    if not isinstance(text, str): # 处理非字符串输入  
        return []  
    text = text.lower() # 转换为小写  
    text = re.sub(r'[^a-z\s]', '', text) # 移除数字和标点，只保留字母和空格  
    tokens = word_tokenize(text) # 分词  
    # 去除停用词并进行词形还原  
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]  
    return processed_tokens  
  
# 应用预处理函数到文本列  
df['processed_tokens'] = df[text_column].apply(preprocess_text)  
# 为了后续 scikit-learn 使用，创建一个处理后连接回字符串的列  
df['processed_text_str'] = df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))  
  
print("\n预处理后的数据:")  
print(df[[text_column, 'processed_tokens', 'processed_text_str']].head())  
  ```

### 方法一：词袋模型 (BoW) 与 TF-IDF

这种方法基于词语的频率来表示文本。

1. 词袋模型 (Bag-of-Words)

BoW 将每个文档表示为一个向量，向量的每个维度代表词汇表中的一个词，其值是该词在文档中出现的次数。
```python
from sklearn.feature_extraction.text import CountVectorizer  
  
# 初始化 CountVectorizer  
# max_features: 限制词汇表大小，只取最常见的 N 个词  
# min_df: 忽略在少于 min_df 个文档中出现的词  
# max_df: 忽略在超过 max_df 比例的文档中出现的词 (例如，0.9 表示忽略在90%以上文档都出现的词)  
bow_vectorizer = CountVectorizer(max_features=100, min_df=1) # 使用较小的 max_features 方便查看  
  
# 对预处理后的文本字符串进行拟合和转换  
bow_matrix = bow_vectorizer.fit_transform(df['processed_text_str'])  
  
# bow_matrix 是一个稀疏矩阵 (scipy.sparse.csr_matrix)  
print("\n词袋模型 (BoW):")  
print(f"特征矩阵形状 (文档数, 词汇表大小): {bow_matrix.shape}")  
  
# 查看词汇表（特征名称）  
print(f"词汇表 (部分): {bow_vectorizer.get_feature_names_out()[:10]}") # 显示前10个词  
  
# 查看第一个文档的BoW向量 (稀疏表示)  
print(f"第一个文档的BoW向量 (稀疏): \n{bow_matrix[0]}")  
# 查看第一个文档的BoW向量 (密集表示)  
# print(f"第一个文档的BoW向量 (密集): \n{bow_matrix[0].toarray()}")  
  ```

2. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF 在 BoW 的基础上，降低了在所有文档中普遍出现的词的权重，提高了在特定文档中重要但在其他文档中少见的词的权重。
```python
from sklearn.feature_extraction.text import TfidfVectorizer  
  
# 初始化 TfidfVectorizer (参数与 CountVectorizer 类似)  
# sublinear_tf=True: 对词频 TF 应用对数平滑 (1 + log(tf))，降低高频词的过度影响  
tfidf_vectorizer = TfidfVectorizer(max_features=100, min_df=1, sublinear_tf=True)  
  
# 对预处理后的文本字符串进行拟合和转换  
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text_str'])  
  
# tfidf_matrix 也是一个稀疏矩阵  
print("\nTF-IDF 模型:")  
print(f"特征矩阵形状 (文档数, 词汇表大小): {tfidf_matrix.shape}")  
  
# 查看词汇表（与BoW可能相同，取决于参数）  
print(f"词汇表 (部分): {tfidf_vectorizer.get_feature_names_out()[:10]}")  
  
# 查看第一个文档的TF-IDF向量 (稀疏表示)  
print(f"第一个文档的TF-IDF向量 (稀疏): \n{tfidf_matrix[0]}")  
# 查看第一个文档的TF-IDF向量 (密集表示)  
# print(f"第一个文档的TF-IDF向量 (密集): \n{tfidf_matrix[0].toarray()}")  
  ```

BoW/TF-IDF 小结:

- 优点: 实现简单，解释性相对较好（知道每个维度代表哪个词）。对于某些任务（如文本分类）效果不错。
    
- 缺点:
    

- 忽略词序和语法结构。
    
- 无法捕捉语义相似性（"狗" 和 "犬" 被视为完全不同的词）。
    
- 产生高维稀疏向量，可能导致维度灾难。

### 方法二：词嵌入 (Word Embeddings)

词嵌入将词语映射到低维、稠密的向量空间，使得语义相近的词在向量空间中的距离也相近。

1. 平均词嵌入 (使用预训练模型)

这是一种简单但常用的方法，计算文档中所有词语（在预训练模型词汇表中存在的）的词向量的平均值，作为整个文档的向量表示。
```python
import gensim.downloader as api  
import numpy as np  
# from nltk.tokenize import word_tokenize # 确保已导入  
  
# 加载预训练模型，例如 GloVe (首次运行会下载模型，可能需要时间)  
# 你也可以选择其他模型，如 'word2vec-google-news-300' 或 'fasttext-wiki-news-subwords-300'  
model_name = "glove-wiki-gigaword-100" # 100维 GloVe 向量  
try:  
    print(f"\n正在加载预训练模型 '{model_name}'...")  
    word_embedding_model = api.load(model_name)  
    print("模型加载成功。")  
    vector_size = word_embedding_model.vector_size # 获取向量维度  
except Exception as e:  
    print(f"加载预训练模型失败: {e}。请检查网络连接或模型名称。")  
    word_embedding_model = None  
    vector_size = 100 # 假设一个默认维度以便代码继续  
  
def get_document_vector_mean(tokens, model, num_features):  
    """  
    计算文档的平均词向量。  
  
    Args:  
        tokens (list): 预处理后的词元列表。  
        model: 加载的Gensim词嵌入模型。  
        num_features (int): 词向量的维度。  
  
    Returns:  
        numpy.ndarray: 文档的平均向量。  
    """  
    if model is None:  
        return np.zeros(num_features) # 模型未加载则返回零向量  
  
    feature_vec = np.zeros((num_features,), dtype="float32")  
    nwords = 0.  
    # 获取模型词汇表集合，用于快速查找  
    index2word_set = set(model.index_to_key) if hasattr(model, 'index_to_key') else set(model.key_to_index.keys())  
  
  
    for word in tokens:  
        if word in index2word_set:  
            nwords = nwords + 1.  
            try:  
                feature_vec = np.add(feature_vec, model[word])  
            except KeyError:  
                # 一般不会发生，因为上面已经检查过 word in index2word_set  
                pass  
  
    if nwords > 0:  
        feature_vec = np.divide(feature_vec, nwords)  
    return feature_vec  
  
# 计算每个文档的平均词向量  
# 注意：这里我们使用之前预处理好的 'processed_tokens' 列  
if 'processed_tokens' in df.columns:  
    df['avg_embedding_vector'] = df['processed_tokens'].apply(  
        lambda tokens: get_document_vector_mean(tokens, word_embedding_model, vector_size)  
    )  
  
    print("\n平均词嵌入向量 (只显示前3个维度):")  
    # 可以将向量展开为DataFrame列  
    embedding_features_df = pd.DataFrame(df['avg_embedding_vector'].to_list(),  
                                        columns=[f'embed_{i}' for i in range(vector_size)],  
                                        index=df.index)  
    print(embedding_features_df.head().iloc[:, :3])  
    print(f"嵌入向量维度: {vector_size}")  
else:  
    print("错误：'processed_tokens' 列不存在，无法计算平均词嵌入。")  
  ```
  

2. 句子/文档嵌入 (使用 Sentence Transformers - 更高级)

像 BERT 这样的模型可以生成考虑上下文的嵌入，通常能更好地捕捉句子或文档的整体语义。sentence-transformers 库简化了这个过程。
```python
# 提示：这部分代码需要安装 `sentence-transformers` 库。  
# 运行模型可能需要较好的硬件和较长时间。  
  
# from sentence_transformers import SentenceTransformer  
  
# # 选择一个预训练的句子转换器模型  
# # 'all-MiniLM-L6-v2' 是一个性能和速度平衡较好的模型  
# # 'paraphrase-multilingual-MiniLM-L12-v2' 支持多种语言  
# sbert_model_name = 'all-MiniLM-L6-v2'  
# try:  
#     print(f"\n正在加载 Sentence Transformer 模型 '{sbert_model_name}'...")  
#     sbert_model = SentenceTransformer(sbert_model_name)  
#     print("模型加载成功。")  
# except Exception as e:  
#     print(f"加载 Sentence Transformer 模型失败: {e}。")  
#     sbert_model = None  
  
# if sbert_model:  
#     # 对原始文本列表进行编码 (通常不需要像BoW那样严格的预处理)  
#     print("正在生成句子嵌入...")  
#     sentence_embeddings = sbert_model.encode(df[text_column].tolist(), show_progress_bar=True)  
  
#     print(f"\n句子嵌入矩阵形状 (文档数, 嵌入维度): {sentence_embeddings.shape}")  
  
#     # 可以将嵌入添加到DataFrame  
#     sbert_features_df = pd.DataFrame(sentence_embeddings,  
#                                      columns=[f'sbert_{i}' for i in range(sentence_embeddings.shape[1])],  
#                                      index=df.index)  
#     print("句子嵌入向量 (只显示前3个维度):")  
#     print(sbert_features_df.head().iloc[:,:3])  
# else:  
#     print("Sentence Transformer 模型未加载，跳过此部分。")  
  
  ```

词嵌入小结:

- 优点:
    

- 能够捕捉词语的语义相似性。
    
- 生成低维、稠密的向量表示，通常更适合深度学习模型。
    
- 预训练模型包含了大量通用语言知识。
    
- 基于 Transformer 的模型能理解上下文。
    

- 缺点:
    

- 简单的平均词嵌入忽略词序。
    
- 需要加载预训练模型，可能占用较多内存。
    
- 基于 Transformer 的模型计算成本较高。
    
- 向量的每个维度缺乏直观的解释性。
    

### 总结与选择

本教程介绍了两种主要的文本向量化方法：

- 词袋模型/TF-IDF: 基于词频的稀疏向量表示，实现简单，适合传统机器学习模型和需要一定解释性的场景。
    
- 词嵌入: 基于分布式表示的稠密向量，能捕捉语义，适合深度学习模型和需要理解词语/句子含义的任务。
    

如何选择？

- 任务需求:
    

- 文本分类、信息检索: BoW/TF-IDF 通常是不错的起点，有时效果就足够好。
    
- 语义相似度计算、问答、机器翻译、情感分析（尤其是细粒度）、聚类: 词嵌入（特别是上下文相关的嵌入）通常表现更优。
    

- 计算资源: BoW/TF-IDF 计算相对较快。词嵌入（尤其是大型预训练模型）需要更多内存和计算时间。
    
- 数据量: 预训练词嵌入模型在数据量较少时也能表现不错，因为它们利用了外部知识。BoW/TF-IDF 在数据量非常大时效果可能也很好。
    
- 模型选择: 传统机器学习模型（如逻辑回归、SVM、朴素贝叶斯）通常与 BoW/TF-IDF 结合使用。深度学习模型（如 CNN, RNN, LSTM, Transformers）更适合使用词嵌入。
    

通常，最佳实践是尝试多种方法，并通过交叉验证来评估哪种向量化技术在你的特定任务和数据集上表现最好。