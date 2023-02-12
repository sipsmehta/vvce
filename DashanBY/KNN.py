# Binary Classification of a disease using K Nearest Neighbors

# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb


# Reading CSV file

# In[3]:


d1=pd.read_csv('column_2C_weka.csv')
d1.head()


# In[4]:


d1.info()


# In[5]:


d1.describe()


# In[6]:


cl=['red' if i=='Abnormal' else 'green' for i in d1.iloc[:,-1]]
pd.plotting.scatter_matrix(d1.iloc[:,:-1],
                           c=cl,
                           figsize=[20,20],
                           diagonal='hist',
                           alpha=0.8,
                           marker='*',
                           s=200,
                           edgecolor='black')
plt.show()


# In[7]:


sb.countplot(x='class',data=d1)
d1['class'].value_counts()


# ##### MODEL

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=3)
X=d1.iloc[:,:-1]
y=d1.iloc[:,-1]
knc.fit(X,y)
pred=knc.predict(X)
print("prediction:{}".format(pred))


# In[9]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=1)
knc.fit(xtrain,ytrain)
pred1=knc.predict(xtest)
from sklearn.metrics import accuracy_score
print("Accuracy:",round(accuracy_score(ytest,pred1)*100,2),"%")# or knc.score(xtest,ytest)


# In[12]:


ne=np.arange(1,25)
train=[]#train accuracy
test=[]#test accuracy
for i,k in enumerate(ne):
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(xtrain,ytrain)
    train.append(knc.score(xtrain, ytrain)*100)
    test.append(knc.score(xtest, ytest)*100)
plt.figure(figsize=[15,10])
plt.plot(ne, test, label = 'Test Accuracy')
plt.plot(ne, train, label = 'Train Accuracy')
plt.legend()
plt.title('Change in Accuracy with change in Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(ne)
plt.show()
print("Best accuracy is {}% with K = {}".format(round(np.max(test),2),test.index(np.max(test))+1))


# DATASET-2



d2=pd.read_csv('column_3C_weka.csv')
d2.head()



d2.info()



d2.describe()



d2['class'].value_counts()




sb.countplot(data=d2,x='class')




X1=d2.iloc[:,:-1]
y1=d2.iloc[:,-1]




X1.head()



y1.head()





from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
transy1=le.fit_transform(y1)




cl1=np.array(['red','yellow','blue']) # Hernia=red , Normal=yellow, Spondylolisthesis=blue
pd.plotting.scatter_matrix(d2.iloc[:,:-1],
                           c=cl1[transy1],
                           figsize=[20,20],
                           diagonal='hist',
                           alpha=0.8,
                           marker='*',
                           s=200,
                           edgecolor='black')
plt.show()





knc=KNeighborsClassifier(n_neighbors=3)
knc.fit(X1,y1)
pred2=knc.predict(X1)
print("prediction:{}".format(pred2))





x1train,x1test,y1train,y1test=train_test_split(X1,y1,test_size=0.3,random_state=1)
knc.fit(x1train,y1train)
pred3=knc.predict(x1test)
print("Accuracy:",round(accuracy_score(y1test,pred3)*100,2),"%")




nr=np.arange(1,25)
train1=[]#train accuracy
test1=[]#test accuracy
for i,k in enumerate(nr):
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(x1train,y1train)
    train1.append(knc.score(x1train, y1train)*100)
    test1.append(knc.score(x1test, y1test)*100)
plt.figure(figsize=[15,10])
plt.plot(nr, test1, label = 'Test Accuracy')
plt.plot(nr, train1, label = 'Train Accuracy')
plt.legend()
plt.title('Change in Accuracy with change in Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(nr)
plt.show()
print("Best accuracy is {}% with K = {}".format(round(np.max(test1),2),test1.index(np.max(test1))+1))
