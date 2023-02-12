import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Bindata = pd.read_csv('column_2C_weka.csv')
MulData = pd.read_csv('column_3C_weka.csv')
Bindata.info()
Bindata.describe()
color_lst = ['red' if i=='Abnormal' else 'green' for i in Bindata.loc[:,'class']]
pd.plotting.scatter_matrix(Bindata.loc[:,Bindata.columns!='class'],c=color_lst,figsize=[20,20],diagonal='hist',alpha=0.5,s=200,marker='o',edgecolor='black')
plt.show()
sns.countplot(x='class',data=Bindata)
Bindata.loc[:,'class'].value_counts()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
x,y = Bindata.loc[:,Bindata.columns!='class'],Bindata.loc[:,'class']
knn.fit(x,y)
pred = knn.predict(x)
print("Prediction {}".format(pred))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors=3)
x,y = Bindata.loc[:,Bindata.columns!='class'],Bindata.loc[:,'class']
Xtrain,Xtest,ytrain, ytest = train_test_split(x,y,test_size=.2)
knn.fit(Xtrain,ytrain)
pred = knn.predict(Xtest)
print("Prediction {}".format(knn.score(Xtest,ytest)))
