#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing important libraries
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataset
data=pd.DataFrame(pd.read_csv('C:\\Users\\mahap\\Downloads\\Iris.csv'))


# In[3]:


#droping the column name id as its not required
data=data.drop(['Id'],axis=1)


# In[4]:



label_encoder = preprocessing.LabelEncoder()


# In[5]:


#using label encoder to make lables for target variable species
data['Species']= label_encoder.fit_transform(data['Species'])
data['Species'].unique()


# In[6]:


data.head(10)


# In[7]:


#checking is there any null values or not
data.isnull().sum()


# In[8]:


data.shape


# In[9]:


sns.set_style('darkgrid')


# In[10]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=data['SepalLengthCm'],y=data['SepalWidthCm'],hue=data['Species'],palette=['green','red','blue'])
plt.title('sepal dimension')
plt.show()


# In[11]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=data['PetalLengthCm'],y=data['PetalWidthCm'],hue=data['Species'],palette=['green','red','blue'])
plt.title('petal dimension')
plt.show()


# In[12]:


#taking the independent variable and target variable
Y=data['Species']
X=data.drop(['Species'],axis=1)


# In[13]:


#using test train split to split the data into training data and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=525)


# In[14]:


clf=DecisionTreeClassifier()


# In[15]:


#training the data
clf=clf.fit(X_train,Y_train)


# In[16]:


clf.get_params()


# In[17]:


#predicting the data 
predictions=clf.predict(X_test)
predictions


# In[18]:


#accuracy of the model
accuracy_score(Y_test,predictions)


# In[19]:


#using confusion matrix to check how many samples are predicted wrong
cf=confusion_matrix(Y_test,predictions,labels=[0,1,2])
cf


# In[20]:


plt.figure(figsize=(12,6))
sns.heatmap(cf, annot=True)
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.title('Heat map for confusion matrix')
plt.show()


# In[21]:


predictions=clf.predict(X_train)
predictions


# In[22]:


dt_feature_names = list(X.columns)
dt_target_names = [str(s) for s in Y.unique()]


# In[23]:


#plotting the tree
fig = plt.figure(figsize=(25,20))
graph= tree.plot_tree(clf, 
                   feature_names=dt_feature_names,  
                   class_names=dt_target_names,
                   filled=True)


# In[24]:


#texual represention of decision tree
text_representation=tree.export_text(clf)

print(text_representation)


# from the above tree you  can see that if petal length less than equal to 2.50 cm then it will be in class 0 it is the root node.IF its greater than 2.50 there are two possibilities , it form a decision node to make a decision for a leaf node.We keep doing this until we find a pure leaf node. Leaf nodes cannot be further divided. And thus a decision tree forms. 
# 
# We can control the depth of the decision tree.  

# In[ ]:




