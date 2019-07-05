
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv("Downloads/Bigmart sales/train.csv")
test=pd.read_csv("Downloads/Bigmart sales/test.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.describe()


# In[6]:


train.info()
train.apply(lambda x: sum(x.isnull()),axis=0)


# In[7]:


train['Outlet_Type'].value_counts()


# In[8]:


train['Outlet_Location_Type'].value_counts()


# In[9]:


train['Outlet_Size'].value_counts()


# In[10]:


train['Item_Identifier'].value_counts()


# In[11]:


train['Outlet_Identifier'].value_counts()


# In[12]:


len(set(train.Outlet_Identifier))


# In[13]:



fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.boxplot(train['Item_Weight'])
plt.show()


# In[14]:


sns.distplot(train.Item_Outlet_Sales, bins = 25)


# In[15]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Item_Outlet_Sales, bins = 25)
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")


# In[16]:


print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())


# In[17]:



#Check for duplicatesidsUnique = 
idsUnique = len(set(train.Item_Identifier))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


# In[18]:


numeric_features = train.select_dtypes(include=['float64','int64'])
numeric_features.dtypes


# In[19]:


corr=numeric_features.corr()
corr
print(corr['Item_Outlet_Sales'].sort_values(ascending=False))


# In[20]:


corr


# In[21]:


#Distribution of variable Item fat content
train['Item_Fat_Content'].value_counts()
sns.countplot(train.Item_Fat_Content)


# In[22]:


sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)


# In[23]:


sns.countplot(train.Item_Type)
plt.xticks(rotation=90)


# In[24]:


plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)


# In[25]:



train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())
x=train.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())


# In[26]:


x


# In[27]:



train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[28]:


train['source']='train'
test['source']='test'
data = pd.concat([train,test], ignore_index = True)
print(train.shape, test.shape, data.shape)


# In[29]:


train.head()


# In[30]:


data.head()


# In[31]:


data.isnull().sum()/data.shape[0]*100


# In[32]:


data.isnull().sum()


# In[33]:


test.shape


# In[34]:


item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')


# In[35]:


item_avg_weight


# In[36]:


data[:][data['Item_Identifier'] == 'DRI11']


# In[37]:


data.Item_Weight.mean()


# In[38]:


data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)
data.isnull().sum()


# In[39]:


outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode


# In[40]:


data['Outlet_Size'].value_counts()


# In[41]:


data['Outlet_Size'].fillna(data['Outlet_Size'].mode(),inplace=True)


# In[42]:


data['Outlet_Size'].isnull().sum()


# In[43]:


data.head(10)


# In[44]:


#replacing the 0 values with the mean value
data['Item_Visibility'].mean()
sum(data['Item_Visibility']==0)
for i in range(14203):
    if data['Item_Visibility'][i]==0:
        data['Item_Visibility'][i]= data['Item_Visibility'].mean()
sum(data['Item_Visibility']==0)


# In[45]:


sum(data['Item_Visibility']==0)


# In[46]:


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
 


# In[47]:


print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull())) 


# In[48]:


data['Outlet_Size'].isnull().sum()


# In[49]:


data.head(20)


# In[50]:


data.Outlet_Size.value_counts()


# In[51]:


data[['Outlet_Size','Outlet_Type']]


# In[52]:



#Remember the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[53]:


data.head()


# In[54]:


data.Outlet_Years


# In[55]:


data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])


# In[56]:


data.head()


# In[57]:


data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[58]:


#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# In[59]:


data.head()


# In[60]:


#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[61]:


data.head()


# In[62]:


del data.Item_Identifier


# In[63]:


def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility


# In[64]:


print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))
 


# In[65]:


func = lambda x: x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]
data['Item_Visibility_MeanRatio'] = data.apply(func,axis=1).astype(float)
data['Item_Visibility_MeanRatio'].describe()


# In[66]:


#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)


# In[67]:


data.shape


# In[68]:


data.head()


# In[71]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Item_Fat_Content','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[72]:


data.head()


# In[77]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [7])
x = onehotencoder.fit_transform(data).toarray()


# In[79]:


#Dummy Variables:
data = pd.get_dummies(data, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet_Identifier'])
data.dtypes


# In[80]:


data.head()


# In[83]:


data.drop(['Item_Identifier'],axis=1,inplace=True)


# In[84]:


data.dtypes


# In[85]:


#Divide into test and train:
traindata = data.loc[data['source']=="train"]
testdata = data.loc[data['source']=="test"]
#Drop unnecessary columns:
testdata.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
traindata.drop(['source'],axis=1,inplace=True)


# In[86]:


traindata.head()


# In[87]:


testdata.head()


# In[136]:


x= traindata.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]].values
y= traindata.iloc[:, 1].values


# In[137]:


xtrain=pd.DataFrame(x)


# In[138]:


ytrain=pd.DataFrame(y)


# In[158]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predicting the Test set results
y_pred = regressor.predict(testdata)


# In[159]:


xtrain.head()


# In[160]:


ytrain.head()


# In[161]:


ytrain.shape


# In[162]:


xtrain.shape


# In[163]:


y_pred


# In[164]:


y_predtrain=regressor.predict(xtrain)


# In[180]:


y_predtrain.shape


# In[166]:


rmse1=np.sqrt(((y_predtrain - ytrain) ** 2).mean())


# rmse1

# In[167]:


rmse1


# In[168]:


# Visualising the Training set results

plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.show()


# In[172]:


rmse1


# In[181]:


# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(xtrain,ytrain)
# Predicting the Test set results
y_pred2 = regressor.predict(testdata)
y_predtrain2=regressor.predict(xtrain)


# In[202]:


y_pred2=y_pred2.reshape((5681,1))


# In[203]:


y_predtrain2=y_predtrain2.reshape((8523,1))


# In[204]:


rmse1=np.sqrt(((y_predtrain - ytrain) ** 2).mean())


# In[205]:


rmse2=np.sqrt(((y_predtrain2 - ytrain) ** 2).mean())


# In[210]:


y_predtrain2


# In[207]:


ytrain.shape


# In[208]:


y_pred2


# In[209]:


rmse2


# In[211]:





# In[212]:


y_pred2=pd.DataFrame(y_pred2)


# In[220]:


y_pred2.to_csv("Downloads/Bigmart sales/Submission.csv",index=False)

