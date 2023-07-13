#!/usr/bin/env python
# coding: utf-8

# ### import libraries

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb 


# ### import data

# In[2]:


df=pd.read_csv(r"D:\job\internship\ineuron\adult.csv\adult.csv")


# ### examining the data

# In[3]:


df.head()


# In[4]:


df.rename(columns = {'education-num':'education_num','marital-status':'marital_status','capital-gain':'capital_gain','capital-loss':'capital_loss'}, inplace = True)


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


for i in df.columns:
    print(i,":\n",df[i].unique())


# In[10]:


df[df == ' ?'] = np.nan


# In[11]:


df.isnull().sum()


# In[12]:


df.shape


# In[13]:


df.dropna(inplace=True)


# In[14]:


df.shape


# ### Visualisation

# In[15]:


sns.heatmap(df.corr(),annot=True)


# Sex

# In[16]:


plt.figure(figsize=(15,10))
ax = sns.countplot(data = df, x = 'sex', hue="salary")

plt.xlabel("Sex")
plt.ylabel("Number of People")
plt.ylim(0,20000)
plt.xticks([0,1],['Male', 'Female'])

for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.16, p.get_height()+1000))

plt.show()


# Workclass

# In[17]:


plt.figure(figsize=(15,6))
sns.histplot(data=df['workclass'], x=df['workclass'], element="bars",kde=True)


# In[18]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['workclass'],hue=df['salary'],palette='viridis',saturation=0.9,edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()


# FnlWgt

# In[19]:


plt.figure(figsize=(10,6))
sns.histplot(x=df['fnlwgt'],hue=df['salary'],bins=15)
plt.tight_layout()
plt.grid(True)
plt.title('Age distribution')
plt.show()


# Education

# In[20]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['education'],palette='husl',hue=df['salary'],saturation=0.9,edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()


# marital-status

# In[21]:


crosstb = pd.crosstab(df["marital_status"], df.salary)
color_palette = ["#B6FFD9","#D9B6FF"] 
barplot = crosstb.plot.bar(rot=0, figsize=(15, 6), color=color_palette)
plt.title("Marital Status vs Salary")
plt.xlabel("Marital Status")
plt.ylabel("Frequency")

plt.show()


# Age

# In[22]:


crosstb = pd.crosstab(df.age, df.salary)
color_palette = ["#B6FFD9","#D9B6FF"]
barplot = crosstb.plot.bar(rot=0,figsize = (15,6),color=color_palette)
plt.xticks(rotation = 90)
plt.show()


# Occupation

# In[23]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['occupation'],palette='viridis',hue=df['salary'],saturation=0.9,edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()


# relationship

# In[24]:


crosstb = pd.crosstab(df.relationship, df.salary)
barplot = crosstb.plot.bar(rot=0,figsize = (15,6),color= ["#C9ADA7","#E5D2C1"])


# race

# In[25]:


plt.figure(figsize=(20,8))
custom_palette = ["#FFB6E3", "#B6FFD9"]
ax = sns.countplot(data = df, x = 'race', hue="salary", palette = custom_palette)

plt.xlabel("Sex", fontsize= 12)
plt.ylabel("Income", fontsize= 12)


for p in ax.patches:
    ax.annotate((p.get_height()), (p.get_x()+0.20, p.get_height()+1000))


# Average hours per week

# In[26]:


plt.figure(figsize=(15,5))
sns.distplot(df['hours-per-week'])
plt.ticklabel_format(style='plain', axis='x') 
plt.show()


# In[27]:


df["hours-per-week"].mean()


# Country

# In[28]:


plt.figure(figsize=(10,6))
sns.countplot(y=df['country'],palette='viridis',saturation=0.9,edgecolor="black")
plt.tight_layout()
plt.grid(True)
plt.show()


# In[29]:


data=[df]


# In[30]:


df.head()


# In[31]:


df['relationship'].unique()


# Data preprocessing

# In[32]:


for dataset in data:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'
    
df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)
df['relationship'] = df['relationship'].map({' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5})

df['marital_status'] = df['marital_status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital_status'] = df['marital_status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
df['marital_status'] = df['marital_status'].map({'Couple':0,'Single':1})

race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}
df['race']= df['race'].map(race_map)

df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)

df.loc[(df['capital_gain'] > 0),'capital_gain'] = 1
df.loc[(df['capital_gain'] == 0 ,'capital_gain')]= 0
df.loc[(df['capital_loss'] > 0),'capital_loss'] = 1
df.loc[(df['capital_loss'] == 0 ,'capital_loss')]= 0

def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': return 'govt'
    elif x['workclass'] == ' Private':return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
    

df['employment_type']=df.apply(f, axis=1)
employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}
df['employment_type'] = df['employment_type'].map(employment_map)
df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)

df['salary']= df['salary'].map({' <=50K':0,' >50K':1})


# In[33]:


df.head()


# In[34]:


df['salary'].unique()

from sklearn.preprocessing import LabelEncoder
for dataset in [df]:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 0
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 1
    dataset.loc[dataset['race'] != ' White', 'race'] = 0
    dataset.loc[dataset['race'] == ' White', 'race'] = 1
    dataset.loc[dataset['workclass'] != ' Private', 'workclass'] = 0
    dataset.loc[dataset['workclass'] == ' Private', 'workclass'] = 1
    dataset.loc[dataset['hours-per-week'] <= 40, 'hours-per-week'] = 0
    dataset.loc[dataset['hours-per-week'] > 40, 'hours-per-week'] = 1
for col in df[df.columns]:
    if df[col].dtypes == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
df = df.astype(int)
df=df.drop(["education"],axis=1)
df.head()
# ### Data preparation

# In[35]:


X= df.drop(['salary'],axis=1)
y=df['salary']
y.value_counts(normalize=True)


# In[36]:


from imblearn.over_sampling import RandomOverSampler
rs = RandomOverSampler(random_state=30)
rs.fit(X,y)


# In[37]:


X_new,y_new = rs.fit_resample(X, y)
y_new.value_counts(normalize=True)


# ### training the dataset

# In[38]:


split_size=0.3

#Creation of Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X_new,y_new,test_size=split_size,random_state=22)

#Creation of Train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)


# In[39]:


print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Validation dataset: {0}{1}".format(X_val.shape, y_val.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))


# ### Model creation

# In[40]:


models = []
names = ['Random Forest','GaussianNB','DecisionTreeClassifier','Adaboost','Xgboost','MLP']

models.append((RandomForestClassifier(n_estimators=100)))
models.append((GaussianNB()))
models.append((DecisionTreeClassifier()))
models.append((AdaBoostClassifier()))
models.append((xgb.XGBClassifier()))
models.append((MLPClassifier()))


# In[41]:


models


# Cross-vaildating models & choosing best fit.

# In[42]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[43]:


kfold = KFold(n_splits=5)
for i in range(0,len(models)):    
    cv_result = cross_val_score(models[i],X_train,y_train,cv=kfold,scoring='accuracy')
    score=models[i].fit(X_train,y_train)
    prediction = models[i].predict(X_val)
    acc_score = round(accuracy_score(y_val,prediction) ,3)    
    print ('_'*40)
    print ('{0}: {1}'.format(names[i],acc_score))


# In[44]:


rf = RandomForestClassifier(n_estimators=110, random_state=56)
                           
rf.fit(X_train,y_train)
prediction = rf.predict(X_test)
print ('-'*40)
print ('Accuracy score:')
print (accuracy_score(y_test,prediction))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,prediction))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,prediction))


# The highest accuracy score is from random forest algorithm.

# In[45]:


classs=['employment_type','education_num','marital_status','relationship','race','sex','hours-per-week','country']
for i in classs:
    print(i)
    print(df[i].unique())


# ### Prediction for input data

# In[46]:


marital_status = {' Married-AF-spouse': 0, ' Married-civ-spouse': 0, ' Divorced': 1, ' Married-spouse-absent': 1, ' Never-married': 1, ' Separated': 1, ' Widowed': 1}
relationship = {' Unmarried': 0, ' Wife': 1, ' Husband': 2, ' Not-in-family': 3, ' Own-child': 4, ' Other-relative': 5}
race = {' White': 0, ' Amer-Indian-Eskimo': 1, ' Asian-Pac-Islander': 2, ' Black': 3, ' Other': 4}
sex = {' Male': 1, ' Female': 0}
capital_gain = {'>0': 1, '=0': 0}
capital_loss = {'>0': 1, '=0': 0}
country = {'US': 1, 'Non-US': 0}
employment_type = {'govt': 0, 'private': 1, 'self_employed': 2, 'without_pay': 3}

feature_names = ['age', 'fnlwgt', 'education_num', 'marital_status', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours-per-week', 'country', 'employment_type']

input_values = {}
for feature in feature_names:
    value = input(f"Enter value for '{feature}': ")
    if feature == 'marital_status':
        nv = marital_status.get(value, value)
    elif feature == 'relationship':
        nv = relationship.get(value, value)
    elif feature == 'race':
        nv = race.get(value, value)
    elif feature == 'sex':
        nv = sex.get(value, value)
    elif feature == 'capital_gain':
        nv = capital_gain.get(value, value)
    elif feature == 'capital_loss':
        nv = capital_loss.get(value, value)
    elif feature == 'country':
        nv = country.get(value, value)
    elif feature == 'employment_type':
        nv = employment_type.get(value, value)
    else:
        nv = value
    input_values[feature] = [nv]

new_data = pd.DataFrame(input_values)


# In[47]:


new_data.head()


# In[48]:


predictions = rf.predict(new_data)


# In[49]:


if predictions==0:
    print("The person is having a salary <=50K")
else:
    print("The person is having a salary >50K")


# ### GUI 

# In[50]:


get_ipython().system('pip install ipywidgets')


# In[51]:


import pandas as pd
import gradio as gr

marital_status = {' Married-AF-spouse': 0, ' Married-civ-spouse': 0, ' Divorced': 1, ' Married-spouse-absent': 1, ' Never-married': 1, ' Separated': 1, ' Widowed': 1}
relationship = {' Unmarried': 0, ' Wife': 1, ' Husband': 2, ' Not-in-family': 3, ' Own-child': 4, ' Other-relative': 5}
race = {' White': 0, ' Amer-Indian-Eskimo': 1, ' Asian-Pac-Islander': 2, ' Black': 3, ' Other': 4}
sex = {' Male': 1, ' Female': 0}
capital_gain = {'>0': 1, '=0': 0}
capital_loss = {'>0': 1, '=0': 0}
country = {'US': 1, 'Non-US': 0}
employment_type = {'govt': 0, 'private': 1, 'self_employed': 2, 'without_pay': 3}

feature_names = ['age', 'fnlwgt', 'education_num', 'marital_status', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours-per-week', 'country', 'employment_type']

input_components = []
for feature in feature_names:
    if feature == 'marital_status':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(marital_status.keys())))
    elif feature == 'relationship':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(relationship.keys())))
    elif feature == 'race':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(race.keys())))
    elif feature == 'sex':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(sex.keys())))
    elif feature == 'capital_gain':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(capital_gain.keys())))
    elif feature == 'capital_loss':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(capital_loss.keys())))
    elif feature == 'country':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(country.keys())))
    elif feature == 'employment_type':
        input_components.append(gr.inputs.Dropdown(label=feature, choices=list(employment_type.keys())))
    else:
        input_components.append(gr.inputs.Textbox(label=feature))

def salary_prediction(*input_values):
    input_dict = {feature: value for feature, value in zip(feature_names, input_values)}
    input_dict['marital_status'] = marital_status[input_dict['marital_status']]
    input_dict['relationship'] = relationship[input_dict['relationship']]
    input_dict['race'] = race[input_dict['race']]
    input_dict['sex'] = sex[input_dict['sex']]
    input_dict['capital_gain'] = capital_gain[input_dict['capital_gain']]
    input_dict['capital_loss'] = capital_loss[input_dict['capital_loss']]
    input_dict['country'] = country[input_dict['country']]
    input_dict['employment_type'] = employment_type[input_dict['employment_type']]
    
    new_data = pd.DataFrame([input_dict])
    predictions = rf.predict(new_data)
    
    if predictions == 0:
        return "The person is having a salary <=50K"
    else:
        return "The person is having a salary >50K"

iface = gr.Interface(fn=salary_prediction,
                     inputs=input_components,
                     outputs="text",
                     title="Salary Prediction",
                     description="Predict the salary level of an individual",
                     )

iface.launch()

