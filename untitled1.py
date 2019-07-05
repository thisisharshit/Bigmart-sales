# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 01:12:24 2019

@author: harshit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train.head()
train.describe()
train.info()
train.apply(lambda x: sum(x.isnull()),axis=0)

train['Item_Weight'].describe()
train['Item_Weight'].mean()
train['Item_Weight'].mode()
train['Item_Weight'].median()

train['Outlet_Type'].value_counts()

train['Outlet_Location_Type'].value_counts()
train['Outlet_Size'].value_counts()
train['Item_Identifier'].value_counts()
train['Outlet_Identifier'].value_counts()

#making various plots on training data
train['Item_Weight'].hist(bins =20)
plt.hist(train['Item_Weight'],10)
plt.hist(train['Item_Weight'])
plt.hist(train['Item_Visibility'])
train['Item_Visibility'].hist(bins =20)
train['Item_Outlet_Sales'].hist(bins =25)
train.hist(column='Item_Fat_Content',by='Loan_Status',bins =20)
train.hist(column='Item_Fat_Content',by='Item_Visibility',bins =20)
train.hist(column='Item_Fat_Content',by='Item_Visibility')

len(set(train.Item_Idetifier))
len(set(train.Outlet_Type))
len(set(train.Outlet_Identifier))
train.shape[0]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(112)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.boxplot(train['Item_Weight'])
plt.show()

#Distribution of target variable (item outlet sales)
sns.distplot(train.Item_Outlet_Sales, bins = 25)
#or
sns.distplot(train['Item_Outlet_Sales'], bins = 25)

#more fancy than above plot
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7,7))
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")

print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())

#Check for duplicatesidsUnique = 
idsUnique = len(set(train.Item_Identifier))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

numeric_features = train.select_dtypes(include=[np.number])
#or
numeric_features = train.select_dtypes(include=['float64','int64'])
numeric_features.dtypes

#cheking the correlation
corr=numeric_features.corr()
corr
print(corr['Item_Outlet_Sales'].sort_values(ascending=False))
#correlation matrixf, 
ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)

#Distribution of variable Item fat content
train['Item_Fat_Content'].value_counts()
sns.countplot(train.Item_Fat_Content)

sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)

sns.countplot(train.Item_Type)
plt.xticks(rotation=90)

plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)

plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Visibility, train["Item_Outlet_Sales"],'.', alpha = 0.3)

plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Type, train["Item_Outlet_Sales"],'.', alpha = 0.3)
plt.xticks(rotation=90)


Outlet_Establishment_Year_pivot = train.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

Item_Fat_Content_pivot = train.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

Outlet_Identifier_pivot = train.pivot_table(index='Outlet_Identifier', values='Item_Outlet_Sales', aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar',figsize=(12,7))
plt.xlabel('Outlet_Identifier')
plt.ylabel('Item_Outlet_Sales')
plt.title('Impact of Outlet_Identifier on Item_Outlet_Sales')
plt.xticks(rotation=0)
plt.show()

train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())
x=train.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())


#Create source column to later separate the data easily
train['source']='train'
test['source']='test'
data = pd.concat([train,test], ignore_index = True)
print(train.shape, test.shape, data.shape)

#Check the percentage of null values per variable
data.isnull().sum()/data.shape[0]*100 #show values in percentage

#aggfunc is mean by default! Ignores NaN by default
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
data[:][data['Item_Identifier'] == 'DRI11']
data.Item_Weight.mean()
data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)
data.isnull().sum()

#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode

#this thing  is not working 
data['Outlet_Size'].value_counts()
data['Outlet_Size'].fillna(data['Outlet_Size'].mode(),inplace=True)
data['Outlet_Size'].isnull().sum()

#Creates pivot table with Outlet_Type and the mean of #Item_Outlet_Sales. Agg function is by default mean()
data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')
#or
#Creates pivot table with Outlet_Type and the mean of #Item_Outlet_Sales. Agg function is by default mean()
data.pivot_table(values='Item_Outlet_Sales', columns='Outlet_Type') #both are same, but second one gives values in rows

#replacing the 0 values with the mean value
data['Item_Visibility'].mean()
sum(data['Item_Visibility']==0)
for i in range(14203):
    if data['Item_Visibility'][i]==0:
        data['Item_Visibility'][i]= data['Item_Visibility'].mean()
sum(data['Item_Visibility']==0)

       
def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))   
       
       
    def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility
print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = 
data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))
       
       
#Remember the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:

data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


from sklearn.preprocessing import LabelEncoder
var_mod = [']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
