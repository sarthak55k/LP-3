#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# In[84]:


data = [
    ['<21', 'High', 'Male', 'Single', 'No'],
    ['<21', 'High', 'Male', 'Married', 'No'],
    ['21-35', 'High', 'Male', 'Single', 'Yes'],
    ['>35', 'Medium', 'Male', 'Single', 'Yes'],
    ['>35', 'Low', 'Female', 'Single', 'Yes'],
    ['>35', 'Low', 'Female', 'Married', 'No'],
    ['21-35', 'Low', 'Female', 'Married', 'Yes'],
    ['<21', 'Medium', 'Male', 'Single', 'No'],
    ['<21', 'Low', 'Female', 'Married', 'Yes'],
    ['>35', 'Medium', 'Female', 'Single', 'Yes'],
    ['<21', 'Medium', 'Female', 'Married', 'Yes'],
    ['21-35', 'Medium', 'Male', 'Married', 'Yes'],
    ['21-35', 'High', 'Female', 'Single', 'Yes'],
    ['>35', 'Medium', 'Male', 'Married', 'No']
]
columns = ['Age', 'Income', 'Gender', 'Marital Status', 'Buys']
df = pd.DataFrame(data,columns=columns)
# df


# In[85]:


le = LabelEncoder()
for i in range(5):
    df[columns[i]] = le.fit_transform(df[columns[i]])


# In[87]:


clf = DecisionTreeClassifier(max_depth=3,min_samples_split=3)


# In[92]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values


# In[93]:


clf.fit(X,y)


# In[94]:


test = [[1,1,0,0]]
clf.predict(test)


# In[95]:


from sklearn.metrics import accuracy_score
y_preds = clf.predict(X)
accuracy_score(y,y_preds)


# In[98]:


from sklearn import tree
fig = plt.figure(figsize=(10,15))
image = tree.plot_tree(clf)


# In[99]:


class Node:
  def __init__(self, feature, values):
    self.feature = feature
    self.values = values
    self.yes = None
    self.no = None

  def __str__(self):
    return f'Feature: {self.feature}, Values: {self.values}'

class DecisionTree:
  def __gini(self, yes_count, no_count):
    yes_total = yes_count[0] + yes_count[1]
    no_total = no_count[0] + no_count[1]
    gini_yes = 1 - (yes_count[0] / yes_total) ** 2 - (yes_count[1] / yes_total) ** 2
    gini_no = 1 - (no_count[0] / no_total) ** 2 - (no_count[1] / no_total) ** 2
    return (yes_total * gini_yes  + no_total * gini_no) / (yes_total + no_total)
  
  def __get_impurity(self, X, y, values):
    yes_count = [0, 0]
    no_count = [0, 0]
    for i in range(len(X)):
      if X[i] in values:
        if y[i]: yes_count[1] += 1
        else: yes_count[0] += 1
      else:
        if y[i]: no_count[1] += 1
        else: no_count[0] += 1
    return self.__gini(yes_count, no_count)

  def __parse(self, x):
    val = list(bin(x)[2:])
    return [i for i in range(len(val)) if val[i] == '1']

  def __get_feature_impurity(self, X, y):
    values = np.unique(X)
    n = 2 ** len(values) - 1
    best_impurity = 100
    for i in range(1, n):
      idx = self.__parse(i)
      val_subset = values[idx].copy()
      impurity = self.__get_impurity(X, y, val_subset)
      if impurity < best_impurity:
        best_impurity = impurity
        best_values = val_subset
    return val_subset, impurity
  
  def __select_best_feature(self, X, y):
    best_impurity = 100
    for feature in X.columns:
      values, impurity = self.__get_feature_impurity(list(X[feature]), y)
      if impurity < best_impurity:
        best_impurity = impurity
        best_feature = feature
        best_values = values

    return best_feature, best_values, best_impurity

  def __filter_data(self, X, y, feature, values, flag):
    X_filtered = X[X[feature].isin(values)].copy()
    idx = list(X_filtered.index)
    X_filtered = X_filtered.reset_index().drop([feature, 'index'], axis = 1)
    y_filtered = y[idx].copy()
    return X_filtered, y_filtered
  
  def __build_tree(self, X, y, parent_impurity = 100):
    best_feature, best_values, impurity = self.__select_best_feature(X, y)
    if impurity >= parent_impurity: return None
    
    node = Node(best_feature, best_values)
    
    X_yes, y_yes = self.__filter_data(X, y, best_feature, best_values, True)
    X_no, y_no = self.__filter_data(X, y, best_feature, best_values, False)
    
    node.yes = self.__build_tree(X_yes, y_yes, impurity)
    node.no = self.__build_tree(X_no, y_no, impurity)

    if node.yes is None: node.yes = True
    if node.no is None: node.no = False
    return node

  def fit(self, X, y):
    self.tree = self.__build_tree(X, y)

  def __make_prediction(self, x, node):
    if type(node) == bool: return node
    
    value = x[node.feature]
    if value in node.values: node = node.yes
    else: node = node.no
    
    return self.__make_prediction(x, node)

  def predict(self, X):
    preds = []
    for i in range(len(X)):
      preds.append(self.__make_prediction(X.iloc[i], self.tree))
    return np.array(preds)


# In[ ]:


df = pd.read_csv('dataset.csv').drop('ID', axis=1)
df


# In[ ]:


train_df = df.iloc[:-1].copy()
test_df = df.iloc[-1:].copy()


# In[ ]:


X_train, y_train = train_df.drop('Buys', axis = 1), np.array(train_df['Buys']) == 'Yes'


# In[ ]:


X_test = test_df.drop('Buys', axis = 1)


# In[ ]:


clf = DecisionTree()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.predict(test_df)


# In[ ]:




