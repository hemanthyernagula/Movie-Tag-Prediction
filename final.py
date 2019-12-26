from django.shortcuts import render
from django.http import HttpResponse
import os
import re
import nltk

import pickle

import pandas as pd
# import matplotlib.pyplot as plt

# from prettytable import from_html_one
# from PIL import Image
# from tqdm import tqdm
# from zipfile import ZipFile
# from prettytable import PrettyTable
from bs4 import BeautifulSoup
from scipy.sparse import hstack
# from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import Counter # Used to count number of times a word repeated
# from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn import metrics
# from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score,f1_score

pkl_dir = os.path.dirname(__file__)
print('Done with imports')

# nltk.download('stopwords')

# stopwords = stopwords.words('english')
# print('done with stopwords')
# with open(pkl_dir+'/pkls/stopwords.pkl','wb') as f:
#       print('saving stopwords')
#       pickle.dump(stopwords,f)

with open(pkl_dir+'/pkls/stopwords.pkl','rb') as f:
      
      stopwords = pickle.load(f)



# stemmer = SnowballStemmer('english')

# with open(pkl_dir+'/pkls/stemmer.pkl','wb') as f:
#       print('saving stopwords')
#       pickle.dump(stemmer,f)

with open(pkl_dir+'/pkls/stemmer.pkl','rb') as f:
      
      stemmer = pickle.load(f)



# f_name = ['murder',
#  'violence',
#  'flashback',
#  'romantic',
#  'cult',
#  'revenge',
#  'psychedelic',
#  'comedy',
#  'suspenseful',
#  'good_versus_evil',
#  'humor',
#  'satire',
#  'entertaining',
#  'neo_noir',
#  'action',
#  'sadist',
#  'insanity',
#  'tragedy',
#  'fantasy',
#  'paranormal',
#  'boring',
#  'mystery',
#  'horror',
#  'melodrama',
#  'cruelty',
#  'gothic',
#  'dramatic',
#  'dark',
#  'atmospheric',
#  'storytelling',
#  'sci_fi',
#  'psychological',
#  'historical',
#  'absurd',
#  'prank',
#  'sentimental',
#  'philosophical',
#  'avant_garde',
#  'bleak',
#  'depressing',
#  'plot_twist',
#  'alternate_reality',
#  'realism',
#  'cute',
#  'stupid',
#  'intrigue',
#  'pornographic',
#  'home_movie',
#  'haunting',
#  'historical_fiction',
#  'allegory',
#  'adult_comedy',
#  'thought_provoking',
#  'inspiring',
#  'anti_war',
#  'comic',
#  'brainwashing',
#  'alternate_history',
#  'queer',
#  'clever',
#  'claustrophobic',
#  'whimsical',
#  'feel_good',
#  'blaxploitation',
#  'western',
#  'grindhouse_film',
#  'suicidal',
#  'magical_realism',
#  'autobiographical',
#  'christian_film',
#  'non_fiction']

# with open(pkl_dir+'/pkls/tags_.pkl','wb') as f:
#       print('saving f_name')
#       pickle.dump(f_name,f)

with open(pkl_dir+'/pkls/tags_.pkl','rb') as f:
      
      f_name = pickle.load(f)      

#---------------------------------------------------


special_chars = pickle.load(open(pkl_dir+'/pkls/special_chars.pkl','rb'))
special_char = pickle.load(open(pkl_dir+'/pkls/special_char.pkl','rb'))
special_chars_meaning = pickle.load(open(pkl_dir+'/pkls/special_chars_meaning.pkl','rb'))
print('loading tfidf')

#Tfidf Vector Files
tfidf_vect_uni = pickle.load(open(pkl_dir+'/pkls/tfidf_vect_uni.pkl','rb'))
tfidf_vect_bi = pickle.load(open(pkl_dir+'/pkls/tfidf_vect_bi.pkl','rb'))
tfidf_vect_tri = pickle.load(open(pkl_dir+'/pkls/tfidf_vect_tri.pkl','rb'))
tfidf_vect_char3 = pickle.load(open(pkl_dir+'/pkls/tfidf_vect_char3.pkl','rb'))
tfidf_vect_char4 = pickle.load(open(pkl_dir+'/pkls/tfidf_vect_char4.pkl','rb'))
l_model_char = pickle.load(open(pkl_dir+'/pkls/movie_tag_model.pkl','rb'))


def clean_data(string_):

      '''This is a function that removes special characters, apply stemming
            and    convert into lower case and return the cleaned string'''

      test_st = BeautifulSoup(string_).get_text()

      # print('Length of str before', len(test_st))

      nn = []
      # sc_in_str = []
      for j in test_st.split():
            
            word = re.sub(r"won't", "will not", j)
            word = re.sub(r"n\'t", " not", word)
            word = re.sub(r"\'ve", " have", word)
            word = re.sub(r"can\'t", "can not", word)
            word = re.sub(r"\'re", " are", word)
            word = re.sub(r"\'s", " is", word)
            word = re.sub(r"\'d", " would", word)
            word = re.sub(r"\'ll", " will", word)
            word = re.sub(r"\'t", " not", word)
            word = re.sub(r"[^a-z0-9]",' ',word)
            word = re.sub(r"\'m", " am", word)
            word = re.sub(r"  "," ",word)

            for i in special_chars:
                if i in word:
                    try:
                        temp =  word
                        word =  word.replace(i,special_chars_meaning[i])
                        word =  word.lower()
                        # sc_in_str.append(i)

                        # print(temp,j)
                        if temp == word:
                          word = word.replace(i,'')
                    except:
                      pass

            if word not in stopwords:
                  nn.append(stemmer.stem(word))
      
     
      
      # sys.stdout.write('\r')

      return ' '.join(nn)
      


def normal_clean_data(string_):

      '''This is a function that removes special characters, apply stemming
            and    convert into lower case and return the cleaned string'''

      test_st = BeautifulSoup(string_).get_text()

      # print('Length of str before', len(test_st))

      nn = []
      # sc_in_str = []
      for j in test_st.split():
           
            for i in special_char:
                if i in j:
                        j =  j.replace(i,'')
                        j =  j.lower()
                        # sc_in_str.append(i)

                        
            
            nn.append(j)
      

      return ' '.join(nn)
      



def model(summary):
    
    l = [summary]
    # for i in range(1):
    #     l.append(summary)
    # pd.DataFrame(l)   
    # l = l.values() 


    pred = l_model_char.predict_proba(hstack((hstack((tfidf_vect_uni.transform(l),tfidf_vect_bi.transform(l),tfidf_vect_tri.transform(l))),hstack((tfidf_vect_char3.transform(l),tfidf_vect_char4.transform(l))))))
    
    d = dict()
    for i in range(len(f_name)):
        if round(pred[0][i]) == 1:
           
            d[f_name[i]] = int(pred[0][i]*100)
    # print('Before sorting___>>',d)
#     del tfidf_vect_bi,tfidf_vect_uni,tfidf_vect_tri,tfidf_vect_char3,tfidf_vect_char4,l_model_char
    sd = dict(sorted(d.items(),key=lambda x:x[1],reverse=True))
    # print('After sorting__>>',sd)
    
    return sd


if __name__=="__main__":
      summery = input('Enter sumary of movie')
      print(model(summery))
           

