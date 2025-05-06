import pandas as pd
import openpyxl
import re
import string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
my_nltk_path="C:\\Users\\DominicMin\\AppData\\Roaming"
nltk.data.path.append(my_nltk_path)

# Data loading and pre-processing 
raw_data=pd.read_csv("py_Code\\Data\\mbti_1.csv")
for i in raw_data.index:
    temp=raw_data.loc[i,"posts"]
    temp=temp.split("|||")
    raw_data.loc[i,"posts"]=temp
    print(f"String splited:line {i}")


def data_clean(data_item):
# data_item should be a list consists of many strings
    # 1.Convert to lower case
    data_lower=[]
    for i in data_item:
        data_lower.append(i.lower())
    
    # 2.Remove URL
    data_without_url=[]
    for i in data_lower:
        data_without_url.append(
        re.sub(
            pattern=r'http\S+|www\S+|https\S+',
            repl='',
            string=i,
            flags=re.MULTILINE
            )
        )
    
    # 3.Remove punctuation
    data_without_punct=[]
    for i in data_without_url:
        data_without_punct.append(
        re.sub(
            pattern=r'[^a-zA-Z0-9\s]',
            repl='',
            string=i
            )
        )

    # 4.Remoce empty string and whitespace characters

    data_without_whitespace=[i for i in data_without_punct if i.strip()]

    # 5.Tokenization
    data_tokens=[]
    for i in data_without_whitespace:
        token=word_tokenize(i)
        data_tokens.append(token)

    # 6.Remove stop words
    stop_words=set(stopwords.words("english"))
    filtered_tokens=[]
    for i in data_tokens:
        temp=[]
        for j in i:
            if j not in stop_words:
                temp.append(j)
        filtered_tokens.append(temp)
    
    # 7.Lemmatization
    lemmatizer = WordNetLemmatizer()
    final_data=[]
    for i in filtered_tokens:
        temp=[]
        for j in i:
            temp.append(lemmatizer.lemmatize(j))
        final_data.append(temp)

    return final_data

# Clean data
for i in raw_data.index:
    data_item=raw_data.loc[i,"posts"]
    raw_data.loc[i,"posts"]=data_clean(data_item)
    print(f"Data processed:line {i} ")
    cleaned_data=raw_data

# Export result
print("Saving data...")
with open('cleaned_data.pkl','wb') as f:
    pickle.dump(cleaned_data,f)
print("Data saved to pickle.")