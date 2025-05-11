#!/usr/bin/env python
# coding: utf-8

# ### Import modules and load data

# In[ ]:


# Import necessary modules for NLP

# Module to load raw data
import pandas as pd

# Modules for NLP
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
my_nltk_path="Data"
nltk.data.path.append(my_nltk_path)
import textstat
# Modules to read/write external files
import json
import pickle

# Average function
def ave(l):
    return sum(l)/len(l)
MBTI_types = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]
# Data loading and spliting 
raw_data=pd.read_csv("Data\\mbti_1.csv")
for i in raw_data.index:
    temp=raw_data.loc[i,"posts"]
    temp=temp.split("|||")
    raw_data.loc[i,"posts"]=temp


# ### Create a class to clean data

# In[ ]:


class Data_to_Clean:

    # Load the contraction map in class
    with open(file="contractions.json",mode='r',encoding='utf-8') as f:
        contractions_map=json.load(f)
    def __init__(self,source=raw_data):
        #self.data should be ALL THE POSTS, type:pd.Series
        self.data=source
        
    # Remove URL
    def remove_url(self):
        def process_remove_url(post):
            post_without_url=[]
            for sentence in post:
                post_without_url.append(
                re.sub(
                    pattern=r'http\S+|www\S+|https\S+',
                    repl='',
                    string=sentence,
                    flags=re.MULTILINE
                    )
                )
            return post_without_url
        self.data["posts"]=self.data["posts"].apply(process_remove_url)
    
    # Expand contractions
    @staticmethod
    def text_expand(original_string,contraction_mapping=contractions_map):
        # Compile an re pattern
        contractions_pattern = re.compile(
            '({})'.format('|'.join(contraction_mapping.keys())),
            flags=re.IGNORECASE|re.DOTALL
            )
        # Map original string to expanded string
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
    def expand_contractions(self):
        def process_expand_contractions(original_list):
            for idx in range(len(original_list)):
                original_list[idx]=Data_to_Clean.text_expand(original_list[idx])
            return original_list
        self.data["posts"]=self.data["posts"].apply(lambda x:process_expand_contractions(x))

    # Convert to lower case
    def tolower(self):
        def process_tolower(post):
            return [
                sentence.lower() for sentence in post
            ]
        self.data["posts"]=self.data["posts"].apply(process_tolower)

    # Remove punctuations
    def remove_punct(self):
        def process_remove_punct(post):
            post_without_punct=[]
            for sentence in post:
                post_without_punct.append(
                    re.sub(
                    pattern=r'[^a-zA-Z\s]',
                    repl=' ',
                    string=sentence
                    )
                )
            return post_without_punct
        self.data["posts"]=self.data["posts"].apply(process_remove_punct)
    # Remove empty string and whitespace characters
    def remove_whitespace(self):
        def process_remove_whitespace(post):
            return [
                sentence for sentence in post if sentence.strip()
            ]
        self.data["posts"]=self.data["posts"].apply(process_remove_whitespace)

    # Tokenization
    def totokens(self):
        def process_totokens(post):
            post_totokens=[]
            for sentence in post:
                tokens=word_tokenize(sentence)
                post_totokens.append(tokens)
            return post_totokens
        self.data["posts"]=self.data["posts"].apply(process_totokens)
    
    # Remove stopwords
    def remove_stopwords(self):
        def process_remove_stopwords(post):
            stop_words=set(stopwords.words("english"))
            filtered_post=[]
            for sentence in post:
                filtered_sentence=[]
                for word in sentence:
                    if word not in stop_words:
                        filtered_sentence.append(word)
                filtered_post.append(filtered_sentence)
            return filtered_post
        self.data["posts"]=self.data["posts"].apply(process_remove_stopwords)

    # Lemmatization
    def post_lemmatize(self):
        def process_lemmatize(post):
            def get_wordnet_postag(old_postag):
                if old_postag.startswith('J'):  
                    return wordnet.ADJ 
                elif old_postag.startswith('V'):  
                    return wordnet.VERB
                elif old_postag.startswith('N'):  
                    return wordnet.NOUN  
                elif old_postag.startswith('R'):  
                    return wordnet.ADV  # 副词  
                else:  
                    return wordnet.NOUN
            lemmatizer=WordNetLemmatizer()
            lemmatized_post=[]
            for tokens in post:
                lemmatized_tokens=[]
                tagged_tokens=pos_tag(tokens)
                for word,tag in tagged_tokens:
                    lemmatized_tokens.append(lemmatizer.lemmatize(word,get_wordnet_postag(tag)))
                lemmatized_post.append(lemmatized_tokens)
            return lemmatized_post
        self.data["posts"]=self.data["posts"].apply(process_lemmatize)
        
    


# ### Create a class to analysis data

# In[ ]:


class Data_to_Analyze(Data_to_Clean):
    def __init__(self,type,source=raw_data):
        # First construct an object of father class
        super().__init__(source)
        # self.data is of type pd.DataFrame, now specific the type
        self.data=self.data.loc[self.data["type"]==type].reset_index(drop=True)
        # Basic identity analysis
        self.basic_identities=pd.Series({

            "type":type,

            "sentence_quantity":[],
            "ave_sentence_quantity":None,
            
            "word_count":[],
            "ave_word_count":None,
 
            "upper_ratio":[],
            "ave_upper_ratio":None,

            "reading_ease":[],
            "ave_reading_ease":None,
            "GF_index":[],
            "ave_GF_index":None
        })
    # Design various methods to get identity data

    def get_sentence_quantity(self):
        for post in self.data["posts"].values:
            self.basic_identities["sentence_quantity"].append(len(post))
        self.basic_identities["ave_sentence_quantity"]=ave(self.basic_identities["sentence_quantity"])
    
    def get_word_count(self):
        for post in self.data["posts"].values:
            ans=0
            for sentence in post:
                ans+=len(sentence.split(" "))
            self.basic_identities["word_count"].append(ans)
        self.basic_identities["ave_word_count"]=ave(self.basic_identities["word_count"])
 
    def get_upper_ratio(self):
        for post in self.data["posts"].values:
            char_count=0;upper_count=0
            for sentence in post:
                for char in sentence:
                    if char.isalpha():
                        char_count+=1
                        if char.isupper():
                            upper_count+=1
            if char_count!=0:
                self.basic_identities["upper_ratio"].append(upper_count/char_count)
            else:
                continue
        self.basic_identities["ave_upper_ratio"]=ave(self.basic_identities["upper_ratio"])
    
    '''
    def get_sentence_length(self):
        sentence_length=[]
        for post in self.data:
            for sentence in post:
                sentence_length.append(len(sentence))
        self.basic_identities["sentence_length"]=sentence_length
        self.basic_identities["ave_sentence_length"]=ave(self.basic_identities["sentence_length"])
    '''

    def get_readability(self):
        reading_ease=[];GF_idx=[]
        for post in self.data["posts"].values:
            concatenated_post=post[0]
            for idx in range(1,len(post)):
                concatenated_post+=post[idx]
            reading_ease.append(
                textstat.flesch_reading_ease(concatenated_post)
            )
            GF_idx.append(
                textstat.gunning_fog(concatenated_post)
            )
        self.basic_identities["reading_ease"]=reading_ease
        self.basic_identities["ave_reading_ease"]=ave(self.basic_identities["reading_ease"])
        self.basic_identities["GF_index"]=GF_idx
        self.basic_identities["ave_GF_index"]=ave(self.basic_identities["GF_index"])




