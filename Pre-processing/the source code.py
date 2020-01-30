import re
import numpy
import os
import io
import nltk
import string

#open the file
file = open('IR.txt','r')
text = file.read()

#lowcast
text = text.lower()

#Poter's Algorithm
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

sentence = text
sentence = porter.stem(sentence)

#tokenization

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
example_sent = text
  
stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(example_sent) 
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

#Stopword removal

output = [] 
stopwords = [",", ".","'s"]
for w in filtered_sentence: 
    if w not in stopwords: 
        output.append(w) 
  
#save as a .txt file

string = "".join(output)
f = open('the result of document 1.txt','w')
f.write(string)
f.close()v
