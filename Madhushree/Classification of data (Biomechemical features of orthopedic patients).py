
# Did this in upgrad LIVE : 12/02/2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv("column_2c_weka.csv")

data.head()

data.tail()

data1= pd.read_csv("column_3c_weka.csv")
# Reading 2nd csv

data1.head()

data1.tail()

### Supervised Learning
Uses data as labels.EX there are orthopedic patients data that have labesls as normal and abnormal

data.info()


data.describe()

### EDA
We are visualizing the data as a part of EDA

color_list =["red" if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

Color List: If ele is Abnormal it should be red else green
This list is passed in scatter plot for visualization

pd.plotting.scatter_matrix(data.loc[:,data.columns!='class'], 
                           c=color_list,
                           figsize=[20,20],
                           diagonal='hist', # In diagonal elements we need hist
                           alpha=0.5,       # transperancy
                           s=200,           # size of the marker
                           marker='*',
                           edgecolor="black")
plt.show()



sns.countplot(x='class',data=data)
data.loc[:, 'class'].value_counts()

#### Classification
###### KNN :
K-Nearest Normal-Check the to which classification does a new data belong,( we may have no of classes)

import sklearn

from sklearn.neighbors import KNeighborsClassifier 

knn=KNeighborsClassifier(n_neighbors=3)
x,y=data.loc[:,data.columns!='class'],data.loc[:,"class"]
knn.fit(x,y)
prediction=knn.predict(x)
print('Prediction: {}'.format(prediction))


## Spliting of data set


## Train test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_text = train_test_split(x,y,test_size=0.30,random_state=1)

### Data Modeling

knn=KNeighborsClassifier(n_neighbors=3)
x,y=data.loc[:,data.columns!='class'],data.loc[:,"class"]
knn.fit(x_train,y_train)
prediction=knn.predict(x_test) # We predict on the testing data
print('The accuracy with K neighbors is : {}',knn.score(x_test,y_text))















