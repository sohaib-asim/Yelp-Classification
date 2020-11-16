#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem import PorterStemmer 
from nltk import word_tokenize
import time


train3 = pd.read_csv("/Users/sohaib/Desktop/EECS 4412/Project/data 2/train3.csv")
test3 = pd.read_csv("/Users/sohaib/Desktop/EECS 4412/Project/data 2/test3.csv")
stop_words = open("/Users/sohaib/Desktop/EECS 4412/Project/stop_words.lst")
stop = stop_words.read()


# In[2]:


ps = PorterStemmer() 

X = train3["text"]
X_full = X.append(test3['text'], ignore_index=True)


# In[3]:


X_full.tail()


# In[4]:


X_full = X.apply(word_tokenize)
X_full = X.apply(lambda x: ''.join([ps.stem(y) for y in x]))


# In[5]:


X_full.head()


# In[6]:


tfidf = TfidfVectorizer(min_df=10, stop_words='english')


# In[7]:


tfidFit = tfidf.fit(X_full)


# In[8]:


vectXtrain = tfidf.transform(train3["text"])
vectXtest = tfidf.transform(test3["text"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits = 10)

kf.get_n_splits(vectXtrain)


# In[28]:


from sklearn.metrics import accuracy_score

accuracy_scores = 0
yTrain = train3['class'].values


# In[29]:


from sklearn.linear_model import LogisticRegression

LogisticModel = LogisticRegression(random_state=0,max_iter=5000)

for train, test in kf.split(vectXtrain):
    LogisticModel.fit(vectXtrain[train], yTrain[train])
    kPred = LogisticModel.predict(vectXtrain[test])
    yTrue = yTrain[test]
    accuracy_scores += accuracy_score(yTrue, kPred)
    
    
print("Logistic Regression")
print(accuracy_scores/10)


# In[30]:


#from sklearn.naive_bayes import GaussianNB

#NaiveBayesModel = GaussianNB()

#accuracy_scores = 0
#yTrain = train3['class'].values


#for train, test in kf.split(vectXtrain):
#    NaiveBayesModel.fit(vectXtrain[train].toarray(), yTrain[train])
#    kPred = NaiveBayesModel.predict(vectXtrain[test].toarray())
#    yTrue = yTrain[test]
#    accuracy_scores += accuracy_score(yTrue, kPred)
    

#print("Naive Bayes")
#print(accuracy_scores/10)


# In[31]:


#from sklearn.neighbors import KNeighborsClassifier

#Knnmodel = KNeighborsClassifier(n_neighbors=9)

#accuracy_scores = 0
#yTrain = train3['class'].values

#for train, test in kf.split(vectXtrain):
#    Knnmodel.fit(vectXtrain[train], yTrain[train])
#    kPred = Knnmodel.predict(vectXtrain[test])
#    yTrue = yTrain[test]
#    accuracy_scores += accuracy_score(yTrue, kPred)
    

#print("K nearest neighbors")
#print(accuracy_scores/10)


# In[32]:


#from sklearn.neural_network import MLPClassifier

#start = time.process_time()
#NNmodel =  MLPClassifier(alpha=0.001, max_iter=1000)
#print(time.process_time() - start)

#accuracy_scores = 0
#yTrain = train3['class'].values

#for train, test in kf.split(vectXtrain):
#    NNmodel.fit(vectXtrain[train], yTrain[train])
#    kPred = NNmodel.predict(vectXtrain[test])
#    yTrue = yTrain[test]
#    accuracy_scores += accuracy_score(yTrue, kPred)

#print("Neural Network")
#print(accuracy_scores/10)


# In[42]:
#from sklearn.ensemble import RandomForestRegressor
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#yTrain = le.fit_transform(train3['class'].values) 

#rfmodel = RandomForestRegressor(n_estimators = 100, random_state = 0)

#accuracy_scores = 0

#for train, test in kf.split(vectXtrain):
#    rfmodel.fit(vectXtrain[train], yTrain[train])
#    kPred = rfmodel.predict(vectXtrain[test])
#    yTrue = yTrain[test]
#    accuracy_scores += accuracy_score(yTrue, kPred.round(), normalize=False)


#print("Random Forest")
#print(accuracy_scores/10)


# In[54]:

predictions = LogisticModel.predict(vectXtest)


# In[66]:


predData = pd.DataFrame({'REVIEW-ID': test3['ID'],'CLASS': predictions})


# In[67]:


predData.head()

# In[69]:


predData['CLASS'].value_counts().sort_index().plot.bar()


# In[70]:


train3['class'].value_counts().sort_index().plot.bar()


# In[75]:


predData.to_csv('/Users/sohaib/Desktop/EECS 4412/Project/final/prediction.csv', index = False, header=True)


# In[74]:


len(tfidf.get_feature_names())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




