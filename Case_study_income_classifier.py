#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:01:49 2019

@author: manimahesh
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns
#To partition the data
from sklearn.model_selection import train_test_split 
# for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir('Documents/Pandas')
data_income =pd.read_csv('income.csv')
data = data_income.copy()
print(data.info())
print("Data cols with null vals \n",data.isnull().sum())
summary_num = data.describe() # summary of numerical data 
print(summary_num)
summary_cat = data.describe(include = 'O')
print(summary_cat)
data['JobType'].value_counts() # Gives count(freq of occurence) of each category
data['occupation'].value_counts()
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
data = pd.read_csv('income.csv',na_values =' ?')
data.isnull().sum()
missing = data[data.isnull().any(axis=1)]

data2 = data.dropna(axis=0)

correlation = data2.corr() #to see corr bn vars close to 1 more rel close to 0 no corr
data2.columns

gender = pd.crosstab(index = data2['gender'],
                     columns = 'count',
                     normalize = True)

gender_salstat = pd.crosstab( index = data2['gender'],
                             columns = data2['SalStat'],
                             margins = True,
                             normalize = 'index')

salstat = sns.countplot(data2['SalStat'])

sns.distplot(data2['age'], bins = 10 , kde = False)

sns.boxplot('SalStat','age',data = data2)
sns.boxplot('age','SalStat',data = data2)
data2['SalStat']
# Reindexing salstat to 0 or 1
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
#each category type col's category is conv to dummy col with val 0 or 1
new_data = pd.get_dummies(data2, drop_first = True)

columns_list = list(new_data.columns)

features = list(set(columns_list)-set('SalStat'))

print(features)

y = new_data['SalStat'].values
print(y)

x = new_data[features].values
print(x)

# splitting datasets for training and testing

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.3, random_state = 0)
logistic = LogisticRegression()

logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

prediction = logistic.predict(test_x)

# to see if predictions are right on test data

confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction = KNN_classifier.predict(test_x)
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)







