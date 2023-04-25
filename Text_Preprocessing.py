#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('emails.csv')


# In[4]:


df.head()


# In[5]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16'],axis=1)


# In[6]:


df1=df[['text','spam']]


# In[7]:


df1.head()


# # 1. Lowercasing

# In[8]:


df1['text'].str.lower()


# In[9]:


df1.head()


# # 2. Removing HTML tags

# In[10]:


import re
def remove_html_tags(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(r'',text)


# # 3. Removing URL'S

# In[11]:


def remove_url(text):
    patten=re.compile(r'http?:||\s+\www\.\s+')
    return pattern.sub(r'',text)


# There is no  HTML tag and URL's so we don't need to work on it .                       
# 

# # 4. Removing punctuations

# In[12]:


import string
exclude=string.punctuation


# In[13]:


def remove_punctuations(text):
    retext=''
    for char in text:
        if char in exclude:
            retext+=''
        else:
            retext+=char
    return retext


# In[14]:


print(remove_punctuations(df1['text']))


# In[45]:


def remove_pun(text):
    return text.translate(str.maketrans('','',exclude))


# In[46]:


df1['text']=df1['text'].apply(remove_pun)


# In[47]:


df1.shape


# # 5. Chat word treatement

# In[39]:


chat_word=pd.read_csv('Abbreviations and Slang.csv')


# In[40]:


chat_word.tail()


# In[41]:


chat_word.shape


# In[ ]:





# In[ ]:





# # 5. Spelling Correction

# In[42]:


pip install -U textblob


# In[43]:


text='wter si evey whre'
from textblob import TextBlob
textblb=TextBlob(text)
textblb.correct().string


# # 6. Removing stop words

# In[26]:


pip install --user -U nltk


# In[33]:


from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")


# In[48]:


df1['text']


# def remove_stopword(text):
#     newtext=[]
#     stop_words=set(stopwords.word('english'))
#     for word in text:
#         if word in stop_words:
#             newtext.append('')
#         else:
#             newtext.append(word)
#     x=newtext[:]
#     newtext.clear()
#     return "".join(x)
# df1['text'].apply(remove_stopword)

# In[87]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = """This is a sample sentence,
				showing off the stop words filtration."""

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#with no lower case conversion
filtered_sentence = []

for w in word_tokens:
	if w not in stop_words:
		filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)


# # 6. Handling Emoji's

# In[52]:


import re

text = u'This dog \U0001f602'
print(text) 

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
print(emoji_pattern.sub(r'', text))


# In[54]:


import emoji
print(emoji.demojize(u'This dog \U0001f602'))


# In[56]:


# 'Emoji_Dict.p'- download link https://drive.google.com/open?id=1G1vIkkbqPBYPKHcQ8qy0G2zkoab2Qv4v
with open('Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
    return text

text = "I won 🥇 in 🏏"
convert_emojis_to_word(text)


# # 7. Tokenization

# i. Using split function:

# In[62]:


word='My name is Akhand'
word.split()


# ii. Using Regular Expression

# In[63]:


import re
re.compile('[.!?]').split(word)


# iii. Using library NLTK

# In[73]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
text="Nowadays, many texts data contains emojis and emoticons due to fast-growing digital communication.Handling these texts "
word_tokenize(text)


# In[74]:


sent_tokenize(text)


# iv. Using Library SPACY 

# In[80]:


get_ipython().system('pip install -U spacy')


# In[81]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[86]:


import spacy

nlp = spacy.load("en_core_web_sm") #Loading dictionary
doc = nlp("This is a sentence.") #Sentence to document
for token in doc:
    print(token)


# # 9. Stemming

# In[88]:


# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]

for w in words:
	print(w, " : ", ps.stem(w))


# # 10. Tokenization

# Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meanings to one word. 
# Text preprocessing includes both Stemming as well as Lemmatization. Many times people find these two terms confusing. Some treat these two as the same. Actually, lemmatization is preferred over Stemming because lemmatization does morphological analysis of the words.
# 
# 
# Applications of lemmatization are: 
# 
# 1. Used in comprehensive retrieval systems like search engines.
# 2. Used in compact indexing

# In[90]:


nltk.download('wordnet')


# In[91]:


# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))


# In[ ]:




