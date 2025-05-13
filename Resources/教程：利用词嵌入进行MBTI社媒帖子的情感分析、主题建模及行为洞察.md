### 引言

在上一篇教程（或你已完成的步骤）中，我们学习了如何将文本数据转换为词嵌入（Word Embeddings）。词嵌入能够捕捉文本的语义信息，将其表示为稠密的数值向量。这些向量是进行更深层次自然语言处理任务的宝贵输入。

本教程将聚焦于获得词嵌入之后，如何利用它们进行：

1. 情感分析 (Sentiment Analysis): 判断帖子所表达的情感倾向（积极、消极、中性）。
    
2. 主题建模 (Topic Modeling): 发现帖子中讨论的主要话题（使用与传统LDA不同的、基于嵌入的方法）。
    
3. 关联分析: 结合MBTI类型，探索不同性格类型在情感表达和话题偏好上的模式。
    
4. 其他研究方向: 提供更多基于文本分析探索MBTI社媒行为的思路。
    

目标： 通过对MBTI社交媒体帖子的情感和主题进行量化分析，揭示不同MBTI类型用户的行为特征和心理倾向。

### 准备工作

1. 必要的库

确保你已安装以下库：
```python
pip install pandas numpy scikit-learn matplotlib seaborn  
# 用于情感分析 (如果使用VADER等)  
pip install vaderSentiment  
# 用于聚类 (主题建模的一种方式)  
# scikit-learn 已包含 K-Means  
# 用于更高级的主题建模 (可选)  
# pip install bertopic  
  
```
2. 假设的数据

我们假设你已经有了一个 Pandas DataFrame，其中至少包含以下列：

- mbti_type: 用户的MBTI类型 (例如: 'INFJ', 'ENTP')。
    
- original_text: 原始的帖子文本内容。
    
- document_embedding: 每个帖子的数据是一个列表，该列表包含了帖子中每一句话的词嵌入（Numpy Array）。 例如，如果一个帖子有3句话，那么这一行的 document_embedding 值会是 [np.array([...]), np.array([...]), np.array([...])]。
    

示例数据结构 (根据你的描述更新):
```python
import pandas as pd  
import numpy as np # 确保导入 numpy  
  
# 假设的示例数据 (你需要用你的真实数据替换)  
# 假设嵌入向量是5维的  
embedding_dim = 5  
data = {  
    'mbti_type': ['INFJ', 'ENTP', 'INFP', 'ISTJ', 'INFJ'],  
    'original_text': [  
        "Feeling deeply connected today, it's wonderful. Another sentence here.", # INFJ - 2 sentences  
        "Let's debate the pros and cons of this new theory!", # ENTP - 1 sentence  
        "Lost in thought, wondering about the meaning of it all. Creative sparks.", # INFP - 2 sentences  
        "Completed all tasks efficiently. Order is satisfying.", # ISTJ - 2 sentences  
        "Sometimes I feel so much, it's overwhelming but also beautiful. One more thing." # INFJ - 2 sentences  
    ],  
    # document_embedding: 每个元素是一个列表，包含该帖子中每句话的Numpy Array嵌入  
    'document_embedding': [  
        [np.random.rand(embedding_dim), np.random.rand(embedding_dim)], # INFJ post  
        [np.random.rand(embedding_dim)],                                 # ENTP post  
        [np.random.rand(embedding_dim), np.random.rand(embedding_dim)], # INFP post  
        [np.random.rand(embedding_dim), np.random.rand(embedding_dim)], # ISTJ post  
        [np.random.rand(embedding_dim), np.random.rand(embedding_dim)]  # INFJ post  
   ]  
}  
df = pd.DataFrame(data)  
  
print("示例DataFrame:")  
print(df.head())  
print("\n第一个帖子的document_embedding示例:")  
print(df.loc[0, 'document_embedding'])  
print(f"其中第一句话的嵌入类型: {type(df.loc[0, 'document_embedding'][0])}")  
  
```
请确保你的 document_embedding 列中的每个Numpy Array向量维度一致。

### 一、基于词嵌入的情感分析

情感分析旨在识别文本中的情感色彩。虽然词嵌入本身不直接给出情感分数，但它们可以作为强大的特征输入到情感分类模型中。

方法：使用聚合后的文档嵌入作为分类器的输入特征

1. 聚合句子嵌入为文档嵌入:  
    由于每个帖子现在是由一个句子嵌入列表表示的，我们需要先将这些句子嵌入聚合成一个单一的文档级嵌入向量。常用的方法是计算平均值。 
    ```python 
    import numpy as np # 确保导入  
      
    def aggregate_sentence_embeddings(list_of_sentence_embeddings, embedding_dim):  
        """  
        将一个帖子中所有句子的嵌入向量列表聚合成一个单一的文档向量（通过平均）。  
        """  
        if not list_of_sentence_embeddings: # 如果列表为空 (例如帖子没有句子或有效嵌入)  
            return np.zeros(embedding_dim) # 返回一个零向量  
      
        # 确保列表中的每个元素都是numpy array  
        valid_embeddings = [emb for emb in list_of_sentence_embeddings if isinstance(emb, np.ndarray) and emb.shape == (embedding_dim,)]  
      
        if not valid_embeddings:  
            return np.zeros(embedding_dim)  
      valis
        return np.mean(valid_embeddings, axis=0)  
      
    # 假设 embedding_dim 是你嵌入向量的维度，例如 5 (根据你的实际情况修改)  
    # embedding_dim = 5 # 你应该从你的嵌入生成过程中知道这个维度  
    # 或者从数据中动态获取 (如果所有嵌入维度一致)  
    if df['document_embedding'].apply(lambda x: len(x) > 0 and hasattr(x[0], 'shape')).any():  
        first_valid_embedding_list = df['document_embedding'].loc[df['document_embedding'].apply(lambda x: len(x) > 0 and hasattr(x[0], 'shape'))].iloc[0]  
        if first_valid_embedding_list and hasattr(first_valid_embedding_list[0], 'shape'):  
            embedding_dim = first_valid_embedding_list[0].shape[0]  
            print(f"\n自动检测到的嵌入维度为: {embedding_dim}")  
        else:  
            print("警告：无法自动检测嵌入维度，请手动设置 embedding_dim。将使用默认值 5。")  
            embedding_dim = 5 # 设置一个默认值或你已知的维度  
    else:  
        print("警告：document_embedding 列中没有有效的嵌入数据来检测维度。请手动设置 embedding_dim。将使用默认值 5。")  
        embedding_dim = 5  
      
      
    # 应用聚合函数创建新的文档级嵌入列  
    df['aggregated_doc_embedding'] = df['document_embedding'].apply(  
        lambda x: aggregate_sentence_embeddings(x, embedding_dim)  
    )  
      
    print("\n添加聚合后的文档嵌入 ('aggregated_doc_embedding'):")  
    print(df[['original_text', 'aggregated_doc_embedding']].head())  
    print("\n第一个帖子的聚合嵌入示例:")  
    print(df.loc[0, 'aggregated_doc_embedding'])  
      
    ```
2. 准备情感标签 (Sentiment Labels):
    

- 理想情况： 你有一个已经标注了情感（如积极、消极、中性）的数据集。
    
- 替代方案1 (无标注数据)： 使用基于词典的方法（如VADER, SentiWordNet，或中文情感词典如HowNet）对 original_text 列进行情感打分，生成伪标签。
    
- 替代方案2 (迁移学习)： 使用在大型情感数据集上预训练好的情感分析模型，直接对你的文本进行预测。
    

为了本教程的演示，我们先用VADER为英文文本生成情感分数作为示例标签。
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
# import numpy as np # 已在前面导入  
  
analyzer = SentimentIntensityAnalyzer()  
  
def get_vader_sentiment_label(text):  
    vs = analyzer.polarity_scores(str(text))  
    compound = vs['compound']  
    if compound >= 0.05:  
        return 'positive'  
    elif compound <= -0.05:  
        return 'negative'  
    else:  
        return 'neutral'  
  
df['sentiment_label_vader'] = df['original_text'].apply(get_vader_sentiment_label)  
print("\n添加VADER情感标签后:")  
print(df[['original_text', 'sentiment_label_vader']].head())  
  
```
3. 准备特征 (X) 和目标 (y):
    

- 特征 X: 你新创建的 aggregated_doc_embedding 列。
    
- 目标 y: 你准备好的情感标签。
    
```python
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
  
# 将聚合后的文档嵌入列表转换为Numpy数组  
# X_embeddings 现在应该是 (文档数, embedding_dim) 的形状  
X_embeddings = np.array(df['aggregated_doc_embedding'].to_list())  
  
label_encoder = LabelEncoder()  
y_sentiment = label_encoder.fit_transform(df['sentiment_label_vader'])  
  
print(f"\n特征 (聚合后的嵌入向量) 形状: {X_embeddings.shape}")  
print(f"目标 (情感标签) 形状: {y_sentiment.shape}")  
print(f"情感标签类别: {label_encoder.classes_}")  
  
if X_embeddings.shape[0] > 0 and len(np.unique(y_sentiment)) > 1: # 确保有数据和多个类别  
    X_train, X_test, y_train, y_test = train_test_split(  
        X_embeddings, y_sentiment, test_size=0.25, random_state=42, stratify=y_sentiment  
    )  
else:  
    print("错误：数据不足或标签类别单一，无法划分训练集和测试集。")  
    # 可以选择将所有数据作为训练集，或跳过模型训练  
    X_train, X_test, y_train, y_test = X_embeddings, np.array([]), y_sentiment, np.array([])  
  ```
  
  

4. 训练一个简单的分类器:  
    我们可以使用如逻辑回归、支持向量机 (SVM) 或简单的神经网络。  
    ```python
    from sklearn.linear_model import LogisticRegression  
    from sklearn.metrics import classification_report  
      
    sentiment_classifier = LogisticRegression(random_state=42, max_iter=1000)  
    if X_train.shape[0] > 0 and y_train.shape[0] > 0: # 确保训练集不为空  
        try:  
           sentiment_classifier.fit(X_train, y_train)  
            print("\n情感分类器训练完成。")  
      
           if X_test.shape[0] > 0: # 确保测试集不为空  
               y_pred = sentiment_classifier.predict(X_test)  
                print("\n情感分类器在测试集上的表现:")  
                print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))  
            else:  
                print("测试集为空，跳过评估。")  
        except ValueError as e:  
           print(f"训练情感分类器时出错: {e}")  
            print("这可能是由于标签类别过少或数据不平衡导致的。请检查情感标签的分布。")  
    else:  
        print("训练数据不足，跳过情感分类器训练。")  
      
    ```
5. 分析MBTI类型与情感的关系:  
    （这部分代码与之前教程相同，使用VADER的连续分数进行分析）  
    ```python
    df['sentiment_score_vader'] = df['original_text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])  
      
    if 'mbti_type' in df.columns:  
        mbti_sentiment_avg = df.groupby('mbti_type')['sentiment_score_vader'].mean().sort_values(ascending=False)  
        print("\n各MBTI类型的平均情感分数 (VADER compound):")  
        print(mbti_sentiment_avg)  
      
        import matplotlib.pyplot as plt  
        import seaborn as sns  
      
       if not mbti_sentiment_avg.empty:  
           plt.figure(figsize=(12, 7))  
            sns.barplot(x=mbti_sentiment_avg.index, y=mbti_sentiment_avg.values, palette="viridis")  
            plt.title("各MBTI类型的平均情感分数")  
            plt.xlabel("MBTI 类型")  
            plt.ylabel("平均情感复合分数 (VADER)")  
            plt.xticks(rotation=45)  
           plt.tight_layout()  
            plt.show()  
        else:  
            print("无法生成MBTI类型与情感关系图，因为分组数据为空。")  
    else:  
        print("DataFrame中缺少 'mbti_type' 列，无法按类型分析情感。")  
      
    ```

### 二、基于词嵌入的主题建模 (通过聚类)

我们将使用聚合后的文档嵌入 (aggregated_doc_embedding，即 X_embeddings) 进行聚类。

方法：对聚合后的文档嵌入进行K-Means聚类

1. 选择合适的聚类数量 (K):  
    （这部分代码与之前教程类似，但确保使用 X_embeddings 即聚合后的文档嵌入）  
    ```python
    from sklearn.cluster import KMeans  
    from sklearn.metrics import silhouette_score  
    import matplotlib.pyplot as plt  
      
    # X_embeddings 是我们之前准备好的聚合后的文档嵌入  
    if X_embeddings.shape[0] < 2: # 检查样本数量是否足够聚类  
        print("错误：文档嵌入数量过少，无法进行聚类。")  
        optimal_k = 0 # 设置为0，后续步骤会跳过  
    else:  
        print("\n开始为聚类选择合适的K值...")  
        sse = []  
        silhouette_scores = []  
        max_k = min(11, X_embeddings.shape[0])  
        k_range = range(2, max_k)  
      
        if len(k_range) > 0:  
            for k_val in k_range: # 避免与外部的k变量冲突  
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')  
                kmeans.fit(X_embeddings)  
                sse.append(kmeans.inertia_)  
                if k_val > 1 and X_embeddings.shape[0] > k_val :  
                    try:  
                        score = silhouette_score(X_embeddings, kmeans.labels_)  
                        silhouette_scores.append(score)  
                       print(f"K={k_val}, 轮廓系数: {score:.3f}")  
                    except ValueError as e_sil:  
                       print(f"K={k_val}, 计算轮廓系数时出错: {e_sil}")  
                        silhouette_scores.append(-1)  
                else:  
                    silhouette_scores.append(-1)  
      
            if sse:  
                plt.figure(figsize=(10, 5))  
                plt.subplot(1, 2, 1)  
                plt.plot(k_range, sse, marker='o')  
                plt.xlabel("簇的数量 (K)")  
               plt.ylabel("SSE")  
                plt.title("肘部法则选择K")  
      
            if any(s != -1 for s in silhouette_scores):  
                plt.subplot(1, 2, 2)  
                plt.plot(k_range, [s for s in silhouette_scores if s != -1], marker='o')  
                plt.xlabel("簇的数量 (K)")  
                plt.ylabel("轮廓系数")  
                plt.title("轮廓系数选择K")  
                plt.tight_layout()  
                plt.show()  
      
            optimal_k = 4 # 默认值  
            if silhouette_scores and any(s != -1 for s in silhouette_scores):  
                valid_scores = [(k_val,s_val) for k_val,s_val in zip(k_range, silhouette_scores) if s_val != -1]  
                if valid_scores:  
                    optimal_k = max(valid_scores, key=lambda item: item[1])[0]  
            print(f"\n根据图表和指标，建议的K值（主题数）约为: {optimal_k}")  
        else:  
            print("K值范围过小或样本数不足，无法进行K值选择。")  
            optimal_k = min(2, X_embeddings.shape[0] -1) if X_embeddings.shape[0] > 1 else 0 # 调整默认值逻辑  
            if optimal_k <= 0: print("无法确定optimal_k，聚类将跳过。")  
      
      ```
      
    
2. 进行K-Means聚类:  
    （代码与之前教程类似，使用 X_embeddings）  
    ```python
    if X_embeddings.shape[0] >= optimal_k and optimal_k > 0:  
        print(f"\n使用 K={optimal_k} 进行K-Means聚类...")  
        kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')  
        df['topic_cluster_embed'] = kmeans_model.fit_predict(X_embeddings)  
        print("聚类完成，已将簇标签添加到DataFrame ('topic_cluster_embed'列)。")  
        print(df[['original_text', 'topic_cluster_embed']].head())  
    elif optimal_k > 0 :  
        print(f"错误：样本数量 ({X_embeddings.shape[0]}) 少于选择的簇数 ({optimal_k})，无法进行聚类。")  
    else:  
        print("错误：无效的簇数量 optimal_k，跳过聚类。")  
      ```
    
3. 解读聚类结果 (主题):  
    （代码与之前教程相同）  
    ```python
    if 'topic_cluster_embed' in df.columns and optimal_k > 0: # 确保 optimal_k > 0  
        # 确保 optimal_k 与实际聚类数一致  
        num_actual_clusters = df['topic_cluster_embed'].nunique()  
        # 如果 optimal_k 在前面被设为0或不合理的值，这里用实际聚类数  
        current_optimal_k_for_loop = min(optimal_k, num_actual_clusters) if optimal_k > 0 else num_actual_clusters  
      
        for i in range(current_optimal_k_for_loop):  
            print(f"\n--- 主题簇 #{i} ---")  
           cluster_df = df[df['topic_cluster_embed'] == i]  
            if not cluster_df.empty:  
                cluster_posts = cluster_df['original_text'].sample(min(5, len(cluster_df))).tolist()  
               print(f"示例帖子 (最多5条):")  
                for post_text in cluster_posts:  
                    print(f"- {post_text[:100]}...")  
      
                from collections import Counter  
                from nltk.tokenize import word_tokenize  
                from nltk.corpus import stopwords  
                stop_words_eng = set(stopwords.words('english'))  
      
               all_cluster_text = " ".join(cluster_df['original_text'].tolist())  
               tokens = word_tokenize(all_cluster_text.lower())  
                meaningful_tokens = [word for word in tokens if word.isalpha() and word not in stop_words_eng and len(word) > 2]  
                most_common_words = Counter(meaningful_tokens).most_common(10)  
                print(f"关键词 (基于词频): {[word for word, freq in most_common_words]}")  
            else:  
                print(f"主题簇 #{i} 为空。")  
    elif optimal_k <= 0:  
        print("未进行聚类，跳过主题解读。")  
      ```
    
4. 分析MBTI类型与主题的关系:  
    （代码与之前教程相同）  
    ```python
    if 'mbti_type' in df.columns and 'topic_cluster_embed' in df.columns:  
       # 确保 topic_cluster_embed 列存在且有有效值  
        if df['topic_cluster_embed'].notna().any():  
           mbti_topic_distribution = pd.crosstab(df['mbti_type'], df['topic_cluster_embed'], normalize='index')  
            print("\n各MBTI类型在不同主题簇上的分布比例:")  
            print(mbti_topic_distribution)  
      
           if not mbti_topic_distribution.empty:  
               plt.figure(figsize=(12, 8))  
                sns.heatmap(mbti_topic_distribution, annot=True, cmap="Blues", fmt=".2f")  
                plt.title("各MBTI类型的主题簇分布热力图")  
                plt.ylabel("MBTI 类型")  
                plt.xlabel("主题簇 (基于嵌入聚类)")  
               plt.show()  
            else:  
                print("无法生成MBTI与主题关系热力图，因为交叉表为空。")  
        else:  
            print("topic_cluster_embed 列不包含有效数据，无法分析MBTI与主题关系。")  
      
    else:  
        print("DataFrame中缺少 'mbti_type' 或 'topic_cluster_embed' 列，无法按类型分析主题。")  
      
    ```

替代方案：BERTopic

（说明与之前教程相同）

### 三、综合分析：MBTI类型、情感与主题

（这部分代码与之前教程相同，但现在它将使用基于聚合嵌入的主题和情感分数）
```python
if 'mbti_type' in df.columns and \  
  'topic_cluster_embed' in df.columns and \  
  df['topic_cluster_embed'].notna().any() and \  
  'sentiment_score_vader' in df.columns:  
  
    mbti_topic_sentiment = df.groupby(['mbti_type', 'topic_cluster_embed'])['sentiment_score_vader'].mean().unstack()  
    print("\n各MBTI类型在不同主题簇下的平均情感分数:")  
    print(mbti_topic_sentiment)  
  
    if not mbti_topic_sentiment.empty:  
        plt.figure(figsize=(14, 10))  
        sns.heatmap(mbti_topic_sentiment, annot=True, cmap="coolwarm", center=0, fmt=".2f")  
        plt.title("各MBTI类型在不同主题簇下的平均情感分数")  
        plt.ylabel("MBTI 类型")  
        plt.xlabel("主题簇")  
        plt.show()  
    else:  
        print("无法生成 MBTI-主题-情感 热力图，因为分组结果为空。")  
else:  
    print("缺少必要的列（mbti_type, topic_cluster_embed, sentiment_score_vader）或 topic_cluster_embed 无有效数据，无法进行综合分析。")  
  
```
- 话题多样性： （说明与之前教程相同）
    
- 情感波动性： （说明与之前教程相同）
    

### 四、其他研究方向

（内容与之前教程相同）

1. 语言风格分析 (Linguistic Style):
    

- 词汇丰富度
    
- 句子复杂度
    
- LIWC 特征
    
- 可读性指标
    

2. 特定N-gram或短语使用
    
3. 社交网络分析
    
4. 基于文本预测MBTI类型
    

- 重要伦理考量
    

5. 时间序列分析
    
6. 与MBTI理论的印证或挑战
    

### 总结

（内容与之前教程类似，但强调了句子嵌入聚合的步骤）

利用聚合后的文档嵌入进行情感分析和主题建模（通过聚类）为理解MBTI类型在社交媒体上的行为提供了强大的工具。通过结合这些分析维度，并进一步探索其他语言特征，你可以从大量文本数据中挖掘出关于不同性格类型如何表达自我、关注什么以及如何与世界互动的深刻洞见。

记住，数据分析是一个迭代的过程。从初步的发现出发，不断提出新的问题，并尝试用数据来回答它们。祝你在MBTI社媒行为的探索之旅中取得丰硕成果！