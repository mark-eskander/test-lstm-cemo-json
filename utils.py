import nltk
import numpy as np
import string
import re
import ast
from keras.preprocessing.text import Tokenizer       
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# nltk.download('punkt')
# nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

# here i put the directory of the same dictionary but in the main notebook
with open(r'tokenizer_june_demo.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

def lower(text):
    ## we want to split the words of the sentence by split() to work with each word individually
    words = text.split()
    ## we created a new list to save all the lowercase words and we converted it by lower() method
    lower = [word.lower() for word in words]
    ## after finishing we join them back by join() method
    return ' '.join(lower)

def hyperlinks(text):
    ## this pattern follows any url
    pattern = r'http\S+|www\S+'
    ## re.sub() is used for substituting all the links with spaces
    removed = re.sub(pattern, '', text)
    return removed

def remove_large_spaces(text):
    ## this pattern is for tabs
    pattern = r'\s+'
    # Remove tabs using regex substitution with spaces
    removed_spaces = re.sub(pattern, ' ', text)
    ## the strip method is used to remove any leading spaces after substitution
    return removed_spaces.strip()

def remove_stopwords(text):
    # checking if the word in the sentences contain stop words or not and save it
    stopword = nltk.corpus.stopwords.words('english')
    text=' '.join([word for word in text.split() if word not in stopword])
    return text

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def remove_non_word_characters(sentence):
    # Regex pattern to match non-word characters
    pattern = r'\W+'
    # Remove non-word characters using regex substitution with spaces
    cleaned_sentence = re.sub(pattern, ' ', sentence)
    return cleaned_sentence

def remove_numbers(text):
    ## this pattern in special for numbers
    pattern = r'\d+'
    # Remove numbers using regex substitution with spaces
    removed_numbers = re.sub(pattern, '', text)
    return removed_numbers

def remove_html(text):
    html_re = re.compile(r'<.*?>')
    # create regex for html tag
    text = re.sub(html_re, '', text)
    return text

def remove_date_time(text):
    # this patterns match date and time formats
    # Matches MM/DD/YYYY or MM/DD/YY
    date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
     # Matches HH:MM or HH:MMAM/HH:MMPM
    time_pattern = r"\d{1,2}:\d{2}([AP]M)?"
    # Remove date and time patterns from the text
    text_without_date = re.sub(date_pattern, "", text)
    text_without_date_time = re.sub(time_pattern, "", text_without_date)
    return text_without_date_time

def remove_mentions_hashtags(text):
    # Remove mentions
    text_without_mentions = re.sub(r"@\w+", "", text)
    # Remove hashtags
    text_without_mentions_hashtags = re.sub(r"#\w+", "", text_without_mentions)
    return text_without_mentions_hashtags

# print(pad_sequences(loaded_dict.texts_to_sequences(['hello good product']), maxlen=90,padding='post',truncating='post'))

functions=[lower ,hyperlinks ,remove_large_spaces , remove_stopwords , remove_punctuation ,
           remove_non_word_characters , remove_numbers , remove_html , remove_date_time ,
           remove_mentions_hashtags ]
 # the function that make all the pre processing
def cleaned_tokenized(x):
    result = x
    for func in functions:
        result = func(result)
        
    result= nltk.word_tokenize(result)
    

    result=' '.join(result)# to combine the tokens to be a sentence
    
    
    result= loaded_dict.texts_to_sequences([result])
    
    result= pad_sequences(result, maxlen=90,padding='post',truncating='post')
    
    return result
