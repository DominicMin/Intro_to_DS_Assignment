# %% [markdown]
# **Abstract**: This program uses Regular Expression (`re`) and Natural Language Toolkit (`nltk`) to clean raw post data and collect some features of the data. It uses object-oriented programming (OOP) strategy and creates father class `Data_to_Clean` and derived class `Data_to_Analyze` including various methods to clean and analyze data.

# %% [markdown]
# ### Import modules and load data

# %%
# Import necessary modules

# Module to load raw data(CSV file)
import pandas as pd
import numpy as np

# Modules for NLP
import re # Regular Expression
import string
from typing import List
import nltk # Natural Language Toolkit
from nltk.tokenize import word_tokenize # For text tokenization
from nltk.corpus import wordnet # For stopwords removal
# For tokens part-of-speech tagging and lemmatization
from nltk import pos_tag 
from nltk.stem import WordNetLemmatizer
my_nltk_path="Data"
nltk.data.path.append(my_nltk_path)
import textstat # Evaluate text readability
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Evaluate text emotion
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
stop_words=set()
stop_words.update(STOPWORDS)

# Modules to read/write external files,etc.
import json
import pickle
import copy

# Language detection 
import fasttext
lang_model=fasttext.load_model("lid.176.bin")

from tqdm.auto import tqdm
tqdm.pandas()

# Average function
def ave(l):
    return sum(l)/len(l)

# MBTI type dictionary
MBTI_types = [
    'istj', 'isfj', 'infj', 'intj', 
    'istp', 'isfp', 'infp', 'intp', 
    'estp', 'esfp', 'enfp', 'entp', 
    'estj', 'esfj', 'enfj', 'entj'
    ]

# %% [markdown]
# ### Create a class to clean data

# %%
class Data_to_Clean:

    # Load the contraction map in class
    with open(file="contractions.json",mode='r',encoding='utf-8') as f:
        contractions_map=json.load(f)
    def __init__(self,source):
        #self.data should be ALL THE POSTS, type:pd.Series
        self.data=source
    
    # Remove "@Mention" and "#Tag"
    def remove_mention_and_tag(self):
        def process_removal(post):
            post_without_mention=[]
            for sentence in post:
                # Use re to scan and substitute
                post_without_mention.append(
                    re.sub(
                        pattern=r'@\w+|#\w+',
                        repl=' ',
                        string=sentence
                    )
                )
            return post_without_mention
        self.data["posts"]=self.data["posts"].apply(process_removal)
        
    # Remove URL
    def remove_url(self):
        def process_remove_url(post):
            post_without_url=[]
            for sentence in post:
                # Use re to scan and substitute
                post_without_url.append(
                re.sub(
                    pattern=r'http\S+|www\S+|https\S+|\n',
                    repl=' ',
                    string=sentence,
                    flags=re.MULTILINE
                    )
                )
            return post_without_url
        self.data["posts"]=self.data["posts"].apply(process_remove_url)
    
    # Remove emoji
    def remove_emoji(self):
        def process_remove_emoji(post):
            post_without_emoji=[]
            for sentence in post:
                # Use re to scan and substitute
                emoji_pattern=re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters, etc.
        "\U0001f926-\U0001f937"  # Supplemental Symbols and Pictographs
        "\U00010000-\U0010ffff"  # Broader range for some less common emojis
        "]+", flags=re.UNICODE
                )
                post_without_emoji.append(
                    emoji_pattern.sub(
                        repl=' ',
                        string=sentence
                    )
                )
            return post_without_emoji
        self.data["posts"]=self.data["posts"].apply(process_remove_emoji)
    
    # Expand contractions
    @staticmethod
    def text_expand(original_string, contraction_mapping=contractions_map):

        standardized_contraction_map = {k.lower(): v for k, v in contraction_mapping.items()}

        sorted_contractions = sorted(
            standardized_contraction_map.items(),
            key=lambda item: len(item[0]),
            reverse=True
        )

        pattern_parts = []
        for contraction, _ in sorted_contractions:
            pattern_parts.append(r'\b' + re.escape(contraction) + r'\b')

        if not pattern_parts:
            return original_string

        contractions_pattern = re.compile(
            '({})'.format('|'.join(pattern_parts)),
            flags=re.IGNORECASE
        )

        def text_mapping(match_obj):
            old_text = match_obj.group(0)
            new_text = standardized_contraction_map.get(old_text.lower())
            
            if new_text:
                return new_text + " "
            else:
                return old_text + " "
        
        expanded_string = contractions_pattern.sub(
            repl=text_mapping,
            string=original_string
        )
        final_result = expanded_string.strip()
        return final_result
    # Apply the function to dataset
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
            result=[sentence for sentence in post if sentence.strip()]
            return result
        self.data["posts"]=self.data["posts"].apply(process_remove_whitespace)

    # Text tokenization
    def totokens(self):
        def process_totokens(post):
            post_totokens=[]
            for sentence in post:
                tokens=word_tokenize(sentence)
                post_totokens.append(tokens)
                # post_totokens.extend(['@SENTENCE-END'])
            return post_totokens
        # here all posts are flatten
        self.data["posts"]=self.data["posts"].apply(process_totokens)
    
    # Apply N-gram
    def apply_ngram(self,phraser):
        def process_apply_ngram(flatten_post):
            # construct N-gram
            post_with_ngram=list(phraser[flatten_post])
            # rebuild sentence structure
            post_rebuild=[]
            new_sentence=[]
            for word in post_with_ngram:
                if word=='@SENTENCE-END':
                    if new_sentence:
                        post_rebuild.append(new_sentence)
                    new_sentence=[]
                else:
                    new_sentence.append(word)
            if new_sentence:
                post_rebuild.append(new_sentence)
            return post_rebuild
        self.data["posts"]=self.data["posts"].apply(process_apply_ngram)
       
    # Remove stopwords in tokenized text
    def remove_stopwords(self):
        def process_remove_stopwords(post):
            filtered_post=[]
            for sentence in post:
                if isinstance(sentence,list):
                    filtered_sentence=[]
                    for word in sentence:
                        if word not in stop_words:
                            filtered_sentence.append(word)
                    filtered_post.append(filtered_sentence)
                else:
                    if sentence not in stop_words:
                        filtered_post.append(sentence)
            return filtered_post
        self.data["posts"]=self.data["posts"].apply(process_remove_stopwords)

    # Lemmatization
    def post_lemmatize(self):
        def process_lemmatize(post):
            # Convert format of part-of-speech tags
            def get_wordnet_postag(old_postag):
                if old_postag.startswith('J'):  
                    return wordnet.ADJ 
                elif old_postag.startswith('V'):  
                    return wordnet.VERB
                elif old_postag.startswith('N'):  
                    return wordnet.NOUN  
                elif old_postag.startswith('R'):  
                    return wordnet.ADV  
                else:  
                    return wordnet.NOUN
            lemmatizer=WordNetLemmatizer()
            lemmatized_post=[]
            for tokens in post:
                lemmatized_tokens=[]
                # Part of speech tagging
                tagged_tokens=pos_tag(tokens)
                # Lemmatize tokens
                for word,tag in tagged_tokens:
                    lemmatized_tokens.append(lemmatizer.lemmatize(word,get_wordnet_postag(tag)))
                lemmatized_post.append(lemmatized_tokens)
            return lemmatized_post
        self.data["posts"]=self.data["posts"].apply(process_lemmatize)

    def concatenate_post(self):
        def process_concatenate_post(post):
            complete_post=[]
            for sentence in post:
                if sentence:
                    complete_post.extend(sentence)
            return complete_post
        self.data["posts"]=self.data["posts"].apply(process_concatenate_post)
        
    

# %% [markdown]
# ### Create a derived class to analysis data

# %%
class Data_to_Analyze(Data_to_Clean):
    def __init__(self,type,source):
        # First initialize an object of father class(Data_to_Clean)
        super().__init__(source)
        # self.data is of type pd.DataFrame, now specific the MBTI type
        self.data=self.data.loc[self.data["type"]==type].reset_index(drop=True)
        # Store bacic identities of the text
        self.basic_identities=pd.Series({

            "type":type,
            # Number of sentences in a post
            "sentence_quantity":[],
            "ave_sentence_quantity":None,
            # Number of words in a post
            "word_count":[],
            "ave_word_count":None,
            # Ratio of upper case characters in a post
            "upper_ratio":[],
            "ave_upper_ratio":None,
            # Two indicators of text readability: Flesch Reading Ease and Gunning Fog Index 
            "reading_ease":[],
            "ave_reading_ease":None,
            "GF_index":[],
            "ave_GF_index":None,
            # Overall text emotion indicator
            "overall_vader_score":None
        })
        self.locations=None

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
    @staticmethod
    def concatenate_full_post(post):
                filtered_post=[sentence for sentence in post if not sentence.isspace()]
                return "".join(filtered_post)
    def get_vader_score(self):
        analyzer = SentimentIntensityAnalyzer()
        overall_vader_score={'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        def addup_score_dict(new_dict,base_dict):
            for key in base_dict.keys():
                base_dict[key]+=new_dict[key]
        def ave_score_dict(base_dict,n):
            for key in base_dict.keys():
                base_dict[key]/=n
        def process_vader_score(post):
            post_vader_score={'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
            for sentence in post:
               addup_score_dict(analyzer.polarity_scores(sentence),base_dict=post_vader_score) 
            ave_score_dict(base_dict=post_vader_score,n=len(post))
            addup_score_dict(new_dict=post_vader_score,base_dict=overall_vader_score)
            return post_vader_score
        self.data["vader_score"]=self.data["posts"].apply(process_vader_score)
        ave_score_dict(overall_vader_score,len(self.data["posts"]))
        self.basic_identities["overall_vader_score"]=overall_vader_score
    

    # Locate string to translate
    def locate_str_to_translate(self):
        result=[]
        for i in tqdm(self.data.index,desc="Locating strings of other languages..."):
            for j in range(len(self.data.loc[i,"posts"])):
                sentence=self.data.loc[i,"posts"][j]
                if len(sentence.split())> 8:
                    lang_prediction=lang_model.predict(
                        re.sub(r'\s+', ' ', sentence).strip(),
                        k=1
                    )
                    if lang_prediction[0][0]!="__label__en" and lang_prediction[1][0]>0.98:
                        result.append((i,j,re.sub(
                            pattern=r"__\w+__",
                            repl='',
                            string=lang_prediction[0][0]
                        )))
        self.locations=result
        with open(f"Data/{self.basic_identities["type"]}_translate_location.pkl","wb") as f:
            pickle.dump(result,f)
    

    def drop_non_english(self,level):
        '''
        To enhance prediction accuracy:
        - sentence must be long
        - remain English punctuation
        - try to convert to lower case and try again
        '''
        def process_drop(post):
            filtered_post=[]
            for sentence in post:
                normalized_sentence=re.sub(r'\s+', ' ', sentence)
                if len(normalized_sentence.split())<6:
                    filtered_post.append(normalized_sentence)
                    # for very short sentence we won't predict
                else:
                    lang=lang_model.predict(normalized_sentence)
                    if lang[0][0]=='__label__en' and lang[1][0]>level:
                        filtered_post.append(normalized_sentence)
                    else:
                        lang=lang_model.predict(normalized_sentence.lower())
                        if lang[0][0]=='__label__en' and lang[1][0]>level:
                            filtered_post.append(normalized_sentence)
            return filtered_post
        self.data["posts"]=self.data["posts"].apply(process_drop)

# %% [markdown]
# ### Construct data clean pipeline 

# %%
def analyze_data_p1(TYPE):
    data=Data_to_Analyze(type=TYPE,source=raw_data)
    data.remove_url()
    data.remove_mention_and_tag()

    # Some features like text readability need to be collected BEFORE the following cleaning procedures
    # Otherwise, they are NOT accurate
    data.get_sentence_quantity()
    data.get_word_count()
    data.get_upper_ratio()
    data.get_readability()
    data.get_vader_score()

    # Continue to clean the data
    data.remove_emoji()
    data.remove_whitespace()
    data.drop_non_english(0.75)
    data.expand_contractions()
    data.tolower()
    data.remove_punct()
    data.remove_whitespace()
    data.totokens()
    data.post_lemmatize()
    
    with open(f"Data\\cleaned_data\\p1\\{TYPE}_cleaned.pkl","wb") as f:
        pickle.dump(data,f)

def analyze_data_p2(TYPE):
    with open(f"Data/cleaned_data/p1/{TYPE}_cleaned.pkl",'rb') as f:
        data=pickle.load(f)
    data.remove_stopwords()
    data.concatenate_post()
    with open(f"Data/cleaned_data/{TYPE}_cleaned.pkl","wb") as f:
        pickle.dump(data,f)

# %% [markdown]
# ### Main execution block

if __name__ == "__main__":
    # Data loading and spliting 
    raw_data=pd.read_csv("Data\\twitter_MBTI.csv",encoding='utf-8')
    raw_data.drop(columns="Unnamed: 0",inplace=True)
    raw_data.columns=["posts","type"]

    # %%
    raw_data.head(20)

    # %%
    raw_data["type"].value_counts()

    # %% [markdown]
    # #### Sentence splitting

    # %%
    for i in raw_data.index:
        raw_data.loc[i,"posts"]=raw_data.loc[i,"posts"].split("|||")

    # Analyze posts from all MBTI types
    for T in tqdm(MBTI_types):
        analyze_data_p1(T)

    # %% [markdown]
    # #### Add custom stopwords

    # %%
    with open("custom_stopwords_new.json","r") as f:
        custom_stopwords=json.load(f)
    stop_words.update(custom_stopwords)
    stop_words

    # %%
    len(stop_words)

    # %%
    for T in tqdm(MBTI_types):
        analyze_data_p2(T)

    # %% [markdown]
    # #### Train N-gram phraser model

    # %%
    # bigarm_model=Phrases(all_posts,min_count=5,threshold=100)
    # bigram_phraser=Phraser(bigarm_model)

    # %%
    # with open("Data/phrases.json","w") as f:
    #     json.dump(bigarm_model.export_phrases(),f)

    # %% [markdown]
    # #### Demonstration of each step

    # %%
    import matplotlib.pyplot as plt
    result={}
    for i in np.arange(0,1,0.15):
        result[f"{i:.2f}"]=0
        infp=Data_to_Analyze('infp',raw_data)
        infp.remove_mention_and_tag()
        infp.remove_emoji()
        infp.remove_url()
        infp.remove_whitespace()
        infp.drop_non_english(i)
        for j in infp.data.index:
            result[f"{i:.2f}"]+=len(infp.data.loc[j,"posts"])
    result=pd.Series(result)
    plt.plot(result.index,result.values,label="Remaining Sentence Quantity")
    plt.xlabel("Filter Level")
    plt.ylabel("Sentence Quantity")
    plt.legend()
    plt.show()

    # %%
    infp=Data_to_Analyze('infp',raw_data)
    infp.remove_url()
    infp.remove_mention_and_tag()
    infp.remove_emoji()
    infp.remove_whitespace()
    infp.remove_punct()
    test=infp.data.loc[0,"posts"]

    # %%
    ans=0
    for sentence in test:
        ans+=len(sentence.split(" "))
    ans

    # %%
    infp.drop_non_english(0.75)
    infp.data.head(30)

    # %%
    infp.expand_contractions()
    infp.data.head(30)

    # %%
    infp.tolower()
    infp.data.head(30)


    # %%
    infp.remove_punct()
    infp.data.head(30)

    # %%
    infp.totokens()
    infp.data.head(30)

    # %%
    infp.post_lemmatize()
    infp.data.head(30)


    # %%
    infp.remove_stopwords()
    infp.data.head(30)

    # %%
    infp.remove_whitespace()
    infp.data.head(30)

    # %% [markdown]
    # ### Consider adding a new dataset

    # %%
    extra_data=pd.read_csv("Data\MBTI_500.csv",encoding='utf-8')
    extra_data

    # %%
    extra_data['posts']=extra_data['posts'].apply(lambda x:x.split())

    # %%
    test=extra_data.loc[0,'posts']
    test

    # %%
    for T in tqdm(MBTI_types):
        data=Data_to_Analyze(type=T,source=extra_data)
        data.remove_stopwords()
        with open(f'Data/cleaned_data/extra/{T}_extra.pkl','wb') as f:
            pickle.dump(data,f)


