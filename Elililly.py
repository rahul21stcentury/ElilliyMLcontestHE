#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from keras import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[2]:


train=pd.read_csv('dataset/train.csv')
test=pd.read_csv('dataset/test.csv')


# In[3]:


train['music_genre'].unique()


# In[4]:


train.head()


# In[5]:


def process(data):
    myli=[]
    data=pd.concat([data,pd.get_dummies(data['key'])],axis=1).drop(['key'],axis=1)
    data=pd.concat([data,pd.get_dummies(data['voice_gender'])],axis=1).drop(['voice_gender'],axis=1)
    data=pd.concat([data,pd.get_dummies(data['mode'])],axis=1).drop(['mode'],axis=1)
    data=pd.concat([data,pd.get_dummies(data['musician_category'])],axis=1).drop(['musician_category'],axis=1)
    if('music_genre' in data.columns):
        le = preprocessing.LabelEncoder()
        genre=le.fit_transform(data['music_genre'])
        data=data.drop(['music_genre'],axis=1)
        genre
        myli.append(genre)

    data=data.drop(['track_name','instance_id'],axis=1)
    data.head()

    data.isnull().sum()

    tempo=data[data['tempo']!='?']['tempo']
    tempo=tempo.astype(float)
    sns.displot(x=tempo ,kind="kde")

    sns.displot(x=data['popularity'] ,kind="kde",color='red')

    sns.displot(x=data['danceability'] ,kind="kde",color='red')

    sns.displot(x=data['duration_ms'] ,kind="kde",color='green')

    data.columns

    for i in data.columns:
        tmp=data[data[i]=='?']
        tmp2=data[data[i]!='?']
        if(tmp.shape[0]>0):
            print(i)
            data=data.replace({i:{'?':tmp2[i].astype('float').mean()}})
        if(data[i].isnull().sum()>0):
            data[i]=data[i].fillna(data[i].mean())

    data.isnull().sum()
    print(data.shape)
    myli.append(data)
    return myli


# In[6]:


td=process(train)
train_data=td[1]
genre=td[0]
test_data=process(test)[0]


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(train_data, genre, test_size=0.33, random_state=42)


# In[9]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
import xgboost as xgb
model=xgb.XGBClassifier(max_depth=15,n_estimators=100)
model.fit(np.asarray(X_train).astype(np.float32),np.asarray(y_train).astype(np.float32))


# In[10]:


res=model.predict(X_test)


# In[11]:


from sklearn.metrics import f1_score
f1_score(res,y_test,average='macro')


# In[12]:


####Running model on test data#########


# In[13]:


res=model.predict(test_data)
res=res.astype(np.int)
res


# In[14]:


df=pd.DataFrame(train['music_genre'])
df['genre']=genre
df=df.drop_duplicates()
genre=list(df['genre'])
genre
music_genre=list(df['music_genre'])
music_genre


# In[15]:


dic={}
for i in range(len(genre)):
    dic[genre[i]]=music_genre[i]
dic


# In[16]:


music_genre=[]
for i in res:
    music_genre.append(dic[i])
music_genre


# In[17]:


op=pd.DataFrame({'instance_id':test['instance_id'],'music_genre':music_genre})
op.to_csv('op.csv',index=False)

