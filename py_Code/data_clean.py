# Import necessary modules for NLP

# Module to load raw data
import pandas as pd

# Modules for NLP
import re
import string
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
my_nltk_path="Data"
nltk.data.path.append(my_nltk_path)

# Modules to read/write external files
import json
import pickle


# Data loading and spliting 
raw_data=pd.read_csv("py_Code\\Data\\mbti_1.csv")
for i in raw_data.index:
    temp=raw_data.loc[i,"posts"]
    temp=temp.split("|||")
    raw_data.loc[i,"posts"]=temp
    print(f"String splited:line {i}")

class Data_to_Clean:

    # Load the contraction map in class
    with open(file="py_Code\\contractions.json",mode='r',encoding='utf-8') as f:
        contractions_map=json.load(f)
    def __init__(self,raw_data,i):
        # raw_data is the original DataFrame,and i=raw_data.index
        self.data=raw_data.loc[i,"posts"]
        # self.data should be a LIST consists of many STRINGS
    
    # Expand contractions
    @staticmethod
    def text_expand(original_string,contraction_mapping=contractions_map):
        #compile an re pattern
        contractions_pattern = re.compile(
            '({})'.format('|'.join(contraction_mapping.keys())),
            flags=re.IGNORECASE|re.DOTALL
            )
        def text_mapping(text_matched):
            old_text=text_matched.group(0)
            new_text=contraction_mapping.get(old_text.lower())
            if not new_text:
                new_text=contraction_mapping.get(old_text)
                if not new_text:
                    return old_text
            return new_text
        expanded_string=contractions_pattern.sub(
            repl=lambda m:text_mapping(m),
            string=original_string
        )
        return expanded_string
    
    # Expand comtractions
    def expand_contractions(self):
        for i in range(len(self.data)):
            self.data[i]=Data_to_Clean.text_expand(self.data[i]) # SOP to use Staticmethod

    # Convert to lower case
    def tolower(self):
        data_lower=[]
        for i in self.data:
            data_lower.append(i.lower())
        self.data=data_lower
    
    # Remove URL
    def remove_url(self):
        data_without_url=[]
        for i in self.data:
            data_without_url.append(
            re.sub(
                pattern=r'http\S+|www\S+|https\S+',
                repl='',
                string=i,
                flags=re.MULTILINE
                )
            )
        self.data=data_without_url
    
    # Remove punctuation
    def remove_punct(self):
        data_without_punct=[]
        for i in self.data:
            data_without_punct.append(
            re.sub(
                pattern=r'[^a-zA-Z0-9\s]',
                repl=' ',
                string=i
                )
            )
        self.data=data_without_punct

    # Remoce empty string and whitespace characters
    def remove_whitespace(self):
        data_without_whitespace=[i for i in self.data if i.strip()]
        self.data=data_without_whitespace

    # Tokenization
    def totokens(self):
        data_tokens=[]
        for i in self.data:
            token=word_tokenize(i)
            data_tokens.append(token)
        self.data=data_tokens

    # Remove stop words
    def remove_stopwords(self):
        stop_words=set(stopwords.words("english"))
        filtered_tokens=[]
        for i in self.data:
            temp=[]
            for j in i:
                if j not in stop_words:
                    temp.append(j)
            filtered_tokens.append(temp)
        self.data=filtered_tokens
    
    # Lemmatization
    def data_lemmatize(self):
        def get_wordnet_pos(old_pos):
            if old_pos.startswith('J'):  
                return wordnet.ADJ 
            elif old_pos.startswith('V'):  
                return wordnet.VERB
            elif old_pos.startswith('N'):  
                return wordnet.NOUN  
            elif old_pos.startswith('R'):  
                return wordnet.ADV  # 副词  
            else:  
                return wordnet.NOUN
        lemmatizer = WordNetLemmatizer()
        lemmatized_data=[]
        for i in range(len(self.data)):
            temp=[]
            tokens=self.data[i]
            tagged_tokens=pos_tag(tokens)
            for word,tag in tagged_tokens:
                temp.append(lemmatizer.lemmatize(word,get_wordnet_pos(tag)))
            lemmatized_data.append(temp)
        self.data=lemmatized_data

def main():
    # Clean the whole DataFrame
    for i in raw_data.index:
        temp=Data_to_Clean(raw_data,i)
        temp.remove_url()
        temp.expand_contractions()
        temp.tolower()
        temp.remove_punct()
        temp.remove_whitespace()
        temp.totokens()
        temp.data_lemmatize()
        #temp.remove_stopwords()
        raw_data.loc[i,"posts"]=temp.data
        print(f"Data cleaned:line {i}")
    cleaned_data=raw_data
    print(cleaned_data.tail())
    # Save the result
    print("Saving result...")
    with open("py_Code\\cleaned_data.pkl","wb") as f:
        pickle.dump(cleaned_data,f)
    print("Result saved successfully!")

if __name__=="__main__":
    main()