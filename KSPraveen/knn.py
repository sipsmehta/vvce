import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('column_2C_weka.csv')
data.head()



data1=pd.read_csv('column_3C_weka.csv')
data1.head()



data.info()


data.describe()

color_list=['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:,data.columns!='class'],c=color_list,figsize=[20,20],diagonal='hist',alpha=0.5,s=200,marker='o',edgecolor="black")
plt.show()

sns.countplot(x='class',data=data)
data.loc[:,'class'].value_counts()

import sklearn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
x,y=data.loc[:,data.columns!='class'],data.loc[:,'class']
knn.fit(x,y)
predection=knn.predict(x)
print('Predection=[]',format(predection))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)

knn=KNeighborsClassifier(n_neighbors=3)
x,y=data.loc[:,data.columns!='class'],data.loc[:,'class']
knn.fit(x_train,y_train)
predection=knn.predict(x_test)
print('The accuracy',knn.score(x_test,y_test))


ne=np.arange(1,25)
train_accuracy=[]
test_accuracy=[]
for i,k in enumerate(ne):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
    
plt.figure(figsize=[13,8])    
plt.plot(ne,test_accuracy,label='Testing accuracy')
plt.plot(ne,train_accuracy,label='Training accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(ne)
plt.savefig('graph.png')
plt.show()
#print("Best accuracy is {} with k = {}",format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
