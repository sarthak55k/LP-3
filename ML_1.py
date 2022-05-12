#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'figure.figsize':(8,8)})


# In[45]:


data = [
    (10,95),
    (9,80),
    (2,10),
    (15,50),
    (10,45),
    (16,98),
    (11,38),
    (16,93)
]


# In[46]:


x=[pt[0] for pt in data]
y=[pt[1] for pt in data]


# In[47]:


sns.scatterplot(x,y,s=100)


# In[48]:


#helper function
def get_mean(val):
    return sum(val)/len(val)

def get_variance(val,mean):
    return sum([(x-mean)**2 for x in val])

def get_covariance(x,mean_x,y,mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x)*(y[i]-mean_y)
    return covar


# In[49]:


def get_coefficients(x,y):
    b1 = 0.0
    b2 = 0.0
    mean_x = get_mean(x)
    mean_y = get_mean(y)
    
    covar_xy = get_covariance(x,mean_x,y,mean_y)
    var_x = get_variance(x,mean_x)
    
    b1 = covar_xy/var_x
    b0 = mean_y - b1*mean_x
    
    return b1,b0


# In[50]:


b1,b0 = get_coefficients(x,y)
print(b1,b0)


# In[53]:


def plot_graph(x,y,slope,intercept):
    axes = sns.scatterplot(x,y,s=100,marker="o")
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', color='red')


# In[54]:


plot_graph(x, y, b1, b0)


# In[55]:


def get_predictions(val):
    return b0 + b1*val

get_predictions(5)


# In[ ]:




