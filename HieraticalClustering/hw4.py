#!/usr/bin/env python
# coding: utf-8

# In[199]:


# IR Programming Assignment4
# 資管三 B06406009 陳姵如
import re
import os
import io
import math
import copy 
import string
import numpy
import sys
from nltk import PorterStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


# In[3]:


# index 1095 documents

RawDocument = []
a = []
b = 0
FileNum = 1095 

for i in range (FileNum):
    b = b+1
    a.append(str(b) + '.txt')


# In[4]:


# Poter's Algorithm

class PorterStemmer:

    def __init__(self):
        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i, j):
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

if __name__ == '__main__':
    p = PorterStemmer()
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            for i in range (FileNum):
                infile = (open(a[i], 'r'))
                tmp = infile
                output = ''
                word = ''
                if tmp == '':
                    break
                for c in tmp:
                    if c.isalpha():
                        word += c.lower()
                    else:
                        if word:
                            output += p.stem(word, 0,len(word)-1)
                            word = ''
                        output += c.lower()
                RawDocument.append(output)
        infile.close()


# In[5]:


# Tokenization

TokenDocument = [] #有重複
Token = []

for i in range (FileNum): 
    TokenDocument.append(word_tokenize(RawDocument[i]))


# In[6]:


# Stopword removal
# Redundancy removal

redundency = ['!','.','?','$',"'s","''",',',"n't","''","'ve",'would','c']
stop_words = set(stopwords.words('english') + redundency)
for i in range (FileNum): 
    tmp = []
    for j in range(len(TokenDocument[i])): 
        if (TokenDocument[i][j] not in stop_words) and (TokenDocument[i][j].isalpha() == True):
            tmp.append(TokenDocument[i][j])
    Token.append(tmp)

# Sort
for i in range(FileNum):  
    Token[i].sort()


# In[7]:


# Establish my dictionary
# Sort my dictionary

Dictionary = []

# for i in range(0, FileNum):
#     text_term[i].sort()
# #print (text_term[0][0])

for i in range (0, FileNum):
    for j in range (len (Token[i]) ):
        if Token[i][j] not in Dictionary:
            Dictionary.append (Token[i][j])
        
Dictionary.sort()    
        
# print(Dictionary)
# print(len(Dictionary))


# In[10]:


# Calculate the df

df = [] #每個詞在幾篇文章出現過

for i in range (len(Dictionary)):
    df.append(0)
    for j in range(FileNum):
        if Dictionary[i] in Token[j]:
            df[i] += 1
            
# print(len(df),df)


# In[11]:


# Calculate the tf
# Calculate the idf
# Calculate the tf-idf

tf = [] #每個詞在“每個文檔”出現幾次
idf = []
tf_idf = []
text_tmp = ""
tf_tmp = []
tf_idf_tmp = []

for i in range(FileNum):
    for j in range (len(Token[i])):
        tf_tmp.append(0)
        text_tmp = Token[i][j]
        for k in range (0, len(Token[i])):
            if (text_tmp == Token[i][k]):
                tf_tmp[j] += 1
    tf.append(tf_tmp)
    tf_tmp = []
    
# print(len(Token[3]))
# print(len(tf[3]),tf[3])

for i in range (len(Dictionary)):
    idf.append(0)
    idf[i] = math.log(FileNum / df[i] , 10)
# print(idf)

for j in range(FileNum):
    for k in range (len(Token[j])):
        for i in range (len(Dictionary)):
            if Token[j][k] == Dictionary[i]:
                tf_idf_tmp.append(tf[j][k] * idf[i])
    tf_idf.append(tf_idf_tmp)
    tf_idf_tmp = []
    
# print(len(Token[1094]), len(tf_idf[1094]))


# In[159]:


# A cosine relation table

vector = [] #一個檔案對應一個向量（維度＝字典詞數）#雙層迴圈 #1095*len(Dictionary)
vector_tmp = []
cosine = [] 
cosine_tmp = []
multi_sum = 0
lenx = 0
leny = 0

for j in range(FileNum):
    for i in range (len(Dictionary)):
        vector_tmp.append(0)
        for k in range (len(Token[j]) ):
            if Dictionary[i] == Token[j][k]:
                vector_tmp[i] = tf_idf[j][k]
    vector.append(vector_tmp)
    vector_tmp = []
# print(vector[595])
# print( len(vector), len(vector[0]), vector[0] )


# In[194]:


vector_tmp = []
for i in range (len(Dictionary)):
    vector_tmp.append(0)
    for k in range (len(Token[595]) ):
        if Dictionary[i] == Token[595][k]:
            vector_tmp[i] = tf_idf[595][k]
vector[595]=vector_tmp
# print(vector[595])


# In[196]:


#Write a function cosine(Docx, Docy) which loads the tf-idf vectors of documents x and y and returns their cosine similarity.

docx = []
docy = []

def cosine(docx, docy): #預設維度一樣
    multi_sum = 0
    lenx = 0
    leny = 0
    cos = 0
    
    for i in range(0, len(docx)):
        multi_sum += docx[i]*docy[i]
        lenx += docx[i]*docx[i]
        leny += docy[i]*docy[i]
    cos = multi_sum / (math.sqrt(lenx) * math.sqrt(leny))
    
    return cos


# In[473]:


#Heap Algorithm
class Heap(object):
    def __init__(self):
        self.__array = []
        self.__last_index = -1
    
    def size(self):
        return len(self.__array)
    
        
    def push(self, value):
        self.__array.append(value)
        self.__last_index += 1
        self.__siftup(self.__last_index)

    def delete(self, index):
        if self.__last_index == -1:
#             raise IndexError("Can't pop from empty heap")
            pass
        root_value = self.__array[index]
        if self.__last_index > 0:  # more than one element in the heap
            self.__array[index] = self.__array[self.__last_index]
            self.__siftdown(index)
        self.__last_index -= 1
        return root_value
        
    def pop(self):
        if self.__last_index == -1:
#             raise IndexError("Can't pop from empty heap")
            pass
        root_value = self.__array[0]
        if self.__last_index > 0:  # more than one element in the heap
            self.__array[0] = self.__array[self.__last_index]
            self.__siftdown(0)
        self.__last_index -= 1
        return root_value

    def peek(self):
        if not self.__array:
            return None
        return self.__array[0]

    def replace(self, new_value):
        if self.__last_index == -1:
#             raise IndexError("Can't pop from empty heap")
            pass
        root_value = self.__array[0]
        self.__array[0] = new_value
        self.__siftdown(0)
        return root_value

    def heapify(self, input_list):
        n = len(input_list)
        self.__array = input_list
        self.__last_index = n-1
        for index in reversed(range(n//2)):
            self.__siftdown(index)

    @classmethod
    def createHeap(cls, input_list):
        heap = cls()
        heap.heapify(input_list)
        return heap

    def __siftdown(self, index):
        current_value = self.__array[index]
        left_child_index, left_child_value = self.__get_left_child(index)
        right_child_index, right_child_value = self.__get_right_child(index)
        best_child_index, best_child_value = (right_child_index, right_child_value) if right_child_index        is not None and self.comparer(right_child_value, left_child_value) else (left_child_index, left_child_value)
        if best_child_index is not None and self.comparer(best_child_value, current_value):
            self.__array[index], self.__array[best_child_index] =                best_child_value, current_value
            self.__siftdown(best_child_index)
        return


    def __siftup(self, index):
        current_value = self.__array[index]
        parent_index, parent_value = self.__get_parent(index)
        if index > 0 and self.comparer(current_value, parent_value):
            self.__array[parent_index], self.__array[index] =                current_value, parent_value
            self.__siftup(parent_index)
        return

    def comparer(self, value1, value2):
        raise NotImplementedError("Should not use the baseclass heap            instead use the class MinHeap or MaxHeap.")

    def __get_parent(self, index):
        
        if index == 0:
            return None, None
        parent_index =  (index - 1) // 2
        return parent_index, self.__array[parent_index]

    def __get_left_child(self, index):
        left_child_index = 2 * index + 1
        if left_child_index > self.__last_index:
            return None, None
        return left_child_index, self.__array[left_child_index]

    def __get_right_child(self, index):
        right_child_index = 2 * index + 2
        if right_child_index > self.__last_index:
            return None, None
        return right_child_index, self.__array[right_child_index]

    def __repr__(self):
        return str(self.__array[:self.__last_index+1])

    def __eq__(self, other):
        if isinstance(other, Heap):
            return self.__array == other.__array
        if isinstance(other, list):
            return self.__array == other
        return NotImplemented

class MaxHeap(Heap):
    def comparer(self, value1, value2):
        return value1 > value2


# In[426]:


# Create a consineTable as a maxheap tree
cosineTable = [] #每個文章存一列
for i in range(FileNum):
    cosineTable_tmp = MaxHeap()
    j=i+1
    for j in range(j, FileNum):
        cosineTable_tmp.push((cosine(vector[i],vector[j]),j))
    cosineTable.append(cosineTable_tmp)
    
# print(cosineTable[1094])


# In[486]:


vector2 = copy.deepcopy(vector) 
cosineTable2 = copy.deepcopy(cosineTable) 


# In[487]:


tmp = []
clustering = [[] for i in range(1095)]


# In[488]:


K20 = 20
docID = -1
peer = -1
clustering_count = 0
merged = []
merged_recorad = []
tmp = []
    
while clustering_count < K20: #收斂到剩下20
    
    maxcos = -1
    
    for i in range(len(cosineTable2)): #len = 1095
        try: #為了解決none
            current = cosineTable2[i].peek()
            if current[1] in merged:
                cosineTable2[i].pop()
#                 print("current_pop",i , current)
            else:
                for i in range(len(cosineTable2)):
                    if current[0] > maxcos: #選出1095個中 cosine similarity 最大
                        maxcos = current[0]
                        docID = i
                        peer = current[1]
        except:
            continue
    
    if docID not in merged:
        merged.append(docID) 
    if peer not in merged:
        merged.append(peer) 
        
    clustering[docID].append(peer) #這行為什麼會填滿全部？
    
    #更新vector[docID]
    for i in range(len(vector2[docID])):
        try:
            vector2[docID][i] = (vector2[docID][i]*(len(clustering[docID])) + vector2[peer][i])/(len(clustering[docID])+1)
        except:
            pass
        
    #更新cosineTable[docID]
    for i in range(cosineTable2[docID].size()):
        k = docID + i + 1
        if(vector[k]): #有些vector被刪掉
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((cosine(vector[docID],vector[k]),k))
        else:
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((-1,k))

        
    #刪除peer vector/clustring
    del cosineTable2[peer] #刪除peer cosinTable
    cosineTable2.insert(peer,tmp)
    del vector2[peer]
    vector2.insert(peer,tmp)
    

    for i in range(len(clustering[peer])-1):
        if clustering[peer][i] not in merged:
            merged.append(clustering[peer][i])
    
    del clustering[peer]
    clustering.insert(peer,tmp)

    clustering_count = 0
    for i in range(len(clustering)):
        if len(clustering[i]) != 0:
            clustering_count = clustering_count + 1
#     print("clustering",clustering)
    print("clustering_count",clustering_count)


# In[ ]:


# print(len(merged),merged)

for i in range (len(clustering)):
    if clustering[i] != []:
        print(i+1)
        temp = set(clustering[i])
        temp = clustering[i]
        temp.sort()
        print(temp)
        print('\n')


# In[ ]:


# Save the 20.txt files

f = open( "20.txt" ,'w')

for i in range(20): 
    if clustering[i] != []:
        f.write ( str(i) + '\n' )
        for j in range ( len(clustering[i]) ):
            f.write(str(clustering[i][j]))

    f.close()


# In[ ]:


vector2 = copy.deepcopy(vector) 
cosineTable2 = copy.deepcopy(cosineTable) 


# In[ ]:


tmp = []
clustering = [[] for i in range(1095)]


# In[ ]:


K13 = 13
docID = -1
peer = -1
clustering_count = 0
merged = []
merged_recorad = []
tmp = []
    
while clustering_count < K13: #收斂到剩下13
    
    maxcos = -1
    
    for i in range(len(cosineTable2)): #len = 1095
        try: #為了解決none
            current = cosineTable2[i].peek()
            if current[1] in merged:
                cosineTable2[i].pop()
#                 print("current_pop",i , current)
            else:
                for i in range(len(cosineTable2)):
                    if current[0] > maxcos: #選出1095個中 cosine similarity 最大
                        maxcos = current[0]
                        docID = i
                        peer = current[1]
        except:
            continue
    
    if docID not in merged:
        merged.append(docID) 
    if peer not in merged:
        merged.append(peer) 
        
    clustering[docID].append(peer) #這行為什麼會填滿全部？
    
    #更新vector[docID]
    for i in range(len(vector2[docID])):
        try:
            vector2[docID][i] = (vector2[docID][i]*(len(clustering[docID])) + vector2[peer][i])/(len(clustering[docID])+1)
        except:
            pass
        
    #更新cosineTable[docID]
    for i in range(cosineTable2[docID].size()):
        k = docID + i + 1
        if(vector[k]!=[]): #有些vector被刪掉
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((cosine(vector[docID],vector[k]),k))
        else:
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((-1,k))

        
    #刪除peer vector/clustring
    del cosineTable2[peer] #刪除peer cosinTable
    cosineTable2.insert(peer,tmp)
    del vector2[peer]
    vector2.insert(peer,tmp)
    

    for i in range(len(clustering[peer])-1):
        if clustering[peer][i] not in merged:
            merged.append(clustering[peer][i])
    
    del clustering[peer]
    clustering.insert(peer,tmp)

    clustering_count = 0
    for i in range(len(clustering)):
        if len(clustering[i]) != 0:
            clustering_count = clustering_count + 1
#     print("clustering",clustering)
    print("clustering_count",clustering_count)


# In[ ]:


for i in range (len(clustering)):
    if clustering[i] != []:
        print(i+1)
        temp = set(clustering[i])
        temp = clustering[i]
        temp.sort()
#         print(temp)
#         print('\n')


# In[ ]:


# Save the 13.txt files

f = open( "13.txt" ,'w')

for i in range(20): 
    if clustering[i] != []:
        f.write ( str(i) + '\n' )
        for j in range ( len(clustering[i]) ):
            f.write(str(clustering[i][j]))

    f.close()


# In[ ]:


vector2 = copy.deepcopy(vector) 
cosineTable2 = copy.deepcopy(cosineTable) 


# In[ ]:


tmp = []
clustering = [[] for i in range(1095)]


# In[ ]:


K8 = 8
docID = -1
peer = -1
clustering_count = 0
merged = []
merged_recorad = []
tmp = []
    
while clustering_count < K8: #收斂到剩下20
    
    maxcos = -1
    
    for i in range(len(cosineTable2)): #len = 1095
        try: #為了解決none
            current = cosineTable2[i].peek()
            if current[1] in merged:
                cosineTable2[i].pop()
#                 print("current_pop",i , current)
            else:
                for i in range(len(cosineTable2)):
                    if current[0] > maxcos: #選出1095個中 cosine similarity 最大
                        maxcos = current[0]
                        docID = i
                        peer = current[1]
        except:
            continue
    
    if docID not in merged:
        merged.append(docID) 
    if peer not in merged:
        merged.append(peer) 
        
    clustering[docID].append(peer) #這行為什麼會填滿全部？
    
    #更新vector[docID]
    for i in range(len(vector2[docID])):
        try:
            vector2[docID][i] = (vector2[docID][i]*(len(clustering[docID])) + vector2[peer][i])/(len(clustering[docID])+1)
        except:
            pass
        
    #更新cosineTable[docID]
    for i in range(cosineTable2[docID].size()):
        k = docID + i + 1
        if(vector[k]!=[]): #有些vector被刪掉
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((cosine(vector[docID],vector[k]),k))
        else:
            cosineTable2[docID].delete(i)
            cosineTable2[docID].push((-1,k))

        
    #刪除peer vector/clustring
    del cosineTable2[peer] #刪除peer cosinTable
    cosineTable2.insert(peer,tmp)
    del vector2[peer]
    vector2.insert(peer,tmp)
    

    for i in range(len(clustering[peer])-1):
        if clustering[peer][i] not in merged:
            merged.append(clustering[peer][i])
    
    del clustering[peer]
    clustering.insert(peer,tmp)

    clustering_count = 0
    for i in range(len(clustering)):
        if len(clustering[i]) != 0:
            clustering_count = clustering_count + 1
#     print("clustering",clustering)
    print("clustering_count",clustering_count)


# In[ ]:


for i in range (len(clustering)):
    if clustering[i] != []:
        print(i+1)
        temp = set(clustering[i])
#         temp = clustering[i]
#         temp.sort()
#         print(temp)
#         print('\n')


# In[ ]:


# Save the 9.txt files

f = open( "8.txt" ,'w')

for i in range(20): 
    if clustering[i] != []:
        f.write ( str(i) + '\n' )
        for j in range ( len(clustering[i]) ):
            f.write(str(clustering[i][j]))

    f.close()


# In[ ]:




