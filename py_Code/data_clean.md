**Abstract**: This program uses Regular Expression (`re`) and Natural Language Toolkit (`nltk`) to clean raw post data and collect some features of the data. It uses object-oriented programming (OOP) strategy and creates father class `Data_to_Clean` and derived class `Data_to_Analyze` including various methods to clean and analyze data.

### Import modules and load data


```python
# Import necessary modules

# Module to load raw data(CSV file)
import pandas as pd

# Modules for NLP
import re # Regular Expression
import string
from typing import List
import nltk # Natural Language Toolkit
from nltk.tokenize import word_tokenize # For text tokenization
from nltk.corpus import stopwords,wordnet # For stopwords removal
# For tokens part-of-speech tagging and lemmatization
from nltk import pos_tag 
from nltk.stem import WordNetLemmatizer
my_nltk_path="Data"
nltk.data.path.append(my_nltk_path)
import textstat # Evaluate text readability
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Evaluate text emotion
# Transformers model to evaluate text emotion
from transformers import pipeline
import torch

# Modules to read/write external files,etc.
import json
import pickle
import copy

# Average function
def ave(l):
    return sum(l)/len(l)

# MBTI type dictionary
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
```


```python
raw_data["type"].value_counts()
```




    type
    INFP    1832
    INFJ    1470
    INTP    1304
    INTJ    1091
    ENTP     685
    ENFP     675
    ISTP     337
    ISFP     271
    ENTJ     231
    ISTJ     205
    ENFJ     190
    ISFJ     166
    ESTP      89
    ESFP      48
    ESFJ      42
    ESTJ      39
    Name: count, dtype: int64



### Create a class to clean data


```python
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
                # Use re to scan and substitute
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
        # Use re.sub() to scan and substitute
        expanded_string=contractions_pattern.sub(
            repl=lambda m:text_mapping(m),
            string=original_string
        )
        return expanded_string
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
            return [
                sentence for sentence in post if sentence.strip()
            ]
        self.data["posts"]=self.data["posts"].apply(process_remove_whitespace)

    # Text tokenization
    def totokens(self):
        def process_totokens(post):
            post_totokens=[]
            for sentence in post:
                tokens=word_tokenize(sentence)
                post_totokens.append(tokens)
            return post_totokens
        self.data["posts"]=self.data["posts"].apply(process_totokens)
    
    # Remove stopwords in tokenized text
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
        
    
```

### Create a derived class to analysis data


```python
class Data_to_Analyze(Data_to_Clean):
    def __init__(self,type,source=raw_data):
        # First initialize an object of father class(Data_to_Clean)
        super().__init__(source)
        # self.data is of type pd.DataFrame, now specific the MBTI type
        self.data=self.data.loc[self.data["type"]==type].reset_index(drop=True)
        self.data_to_vec=None
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
    
    def get_transformer_emotion():
        device=0 if torch.cuda.is_available() else -1
        emotion_pipeline=pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

```

#### Create a function including all the procedures of data 


```python
def analyze_data(TYPE):
    data=Data_to_Analyze(type=TYPE)
    data.remove_url()
    # Some features like text readability need to be collected BEFORE the following cleaning procedures
    # Otherwise, they are NOT accurate
    data.get_vader_score()
    data.get_sentence_quantity()
    data.get_word_count()
    data.get_upper_ratio()
    data.get_readability()
    print(data.basic_identities["type"],":",data.basic_identities["overall_vader_score"])
    # Continue to clean the data
    data.expand_contractions()
    data.tolower()
    data.remove_punct()
    data.remove_whitespace()
    data.totokens()
    data.data_to_vec = copy.deepcopy(data.data)
    data.remove_stopwords()
    data.post_lemmatize()
    # Save cleaned data to pickle binary files so that they can be loaded easily in other programs
    with open(f"Data\\cleaned_data\\{TYPE}_cleaned.pkl","wb") as f:
        pickle.dump(data,f)

# Analyze posts from all MBTI types

for T in MBTI_types:
    analyze_data(T)

# analyze_data("INFP")
```

    ISTJ : {'neg': 0.06462386341771696, 'neu': 0.7691354115172823, 'pos': 0.13866947186418419, 'compound': 0.2076339583522606}
    ISFJ : {'neg': 0.06356502335550579, 'neu': 0.7465097492067994, 'pos': 0.16155458537696593, 'compound': 0.2672021876971982}
    INFJ : {'neg': 0.06537576657278607, 'neu': 0.7567782029328746, 'pos': 0.15076427145316323, 'compound': 0.24372996125794777}
    INTJ : {'neg': 0.068073406834468, 'neu': 0.772312662774562, 'pos': 0.13445292892292166, 'compound': 0.18367194923473032}
    ISTP : {'neg': 0.07226024094075342, 'neu': 0.7621213063727679, 'pos': 0.1325203848947357, 'compound': 0.16335784346594834}
    ISFP : {'neg': 0.06281754749596304, 'neu': 0.7393171430432627, 'pos': 0.15936036668567108, 'compound': 0.25198372446376077}
    INFP : {'neg': 0.0699536057978941, 'neu': 0.7465856270242714, 'pos': 0.1529217288594955, 'compound': 0.2326227567422329}
    INTP : {'neg': 0.06971799933219272, 'neu': 0.7706881588154774, 'pos': 0.1299565986654305, 'compound': 0.17049641116170697}
    ESTP : {'neg': 0.06968665908536865, 'neu': 0.7609431455393987, 'pos': 0.14755981026111775, 'compound': 0.20030727538651671}
    ESFP : {'neg': 0.06429603290407408, 'neu': 0.7546828902775884, 'pos': 0.15634973189411577, 'compound': 0.22672589583406902}
    ENFP : {'neg': 0.06516937032410103, 'neu': 0.7427469802005195, 'pos': 0.17440461676730257, 'compound': 0.30023835487541434}
    ENTP : {'neg': 0.07009561797582091, 'neu': 0.7692918971182391, 'pos': 0.14303599998343414, 'compound': 0.20488882586058915}
    ESTJ : {'neg': 0.06579453032719008, 'neu': 0.7630612693555346, 'pos': 0.15107213798581864, 'compound': 0.22051428050574087}
    ESFJ : {'neg': 0.05926301585543262, 'neu': 0.7632908940671477, 'pos': 0.16728238168597856, 'compound': 0.30349027080226876}
    ENFJ : {'neg': 0.06404391898412923, 'neu': 0.7400019841766289, 'pos': 0.17974621054186013, 'compound': 0.31034735104959865}
    ENTJ : {'neg': 0.0697524142090613, 'neu': 0.7657310873903306, 'pos': 0.14413921754675463, 'compound': 0.20317545824226105}
    


```python
def get_word_count(self):
        for post in self.data["posts"].values:
            ans=0
            for sentence in post:
                ans+=len(sentence.split(" "))
            self.basic_identities["word_count"].append(ans)
        self.basic_identities["ave_word_count"]=ave(self.basic_identities["word_count"])
```

