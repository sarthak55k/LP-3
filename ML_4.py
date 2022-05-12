#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import seaborn as sns
sns.set(rc={'figure.figsize':(7,7)})


# In[14]:


class KMeans:
    def __init__(self,k):
        self.k = k
    
    def __distance(self,x,y):
        return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    
    def fit(self,points,centroids):
        prev_clusters = None
        clusters = [set() for _ in range(self.k)]
        
        while prev_clusters != clusters:
            prev_clusters = clusters
            for p in points:
                idx = 0
                for i in range(1,self.k):
                    if self.__distance(p,centroids[i])<self.__distance(p,centroids[idx]):
                        idx = i
                clusters[idx].add(p)
            for i in range(self.k):
                centroids[i] = np.mean(list(clusters[i]),axis=0)
        return clusters,centroids


# In[15]:


points = [
          (0.1, 0.6),
          (0.15, 0.71),
          (0.08,0.9),
          (0.16, 0.85),
          (0.2,0.3),
          (0.25,0.5),
          (0.24,0.1),
          (0.3,0.2)
]
centroids = [(0.1, 0.6),(0.3,0.2)]


# # Before Clustering

# In[16]:


x = [pt[0] for pt in points]
y = [pt[1] for pt in points]


# In[17]:


sns.scatterplot(x,y)


# # After clustering

# In[18]:


kms = KMeans(2)


# In[19]:


clusters, centroids = kms.fit(points,centroids)


# In[20]:


clustered_df = pd.DataFrame()
x = []
y = []
category = []
for i in range(len(clusters)):
  for p in clusters[i]:
    x.append(p[0])
    y.append(p[1])
    category.append(f'{i}')
for c in centroids:
  x.append(c[0])
  y.append(c[1])
  category.append('Centroid')
clustered_df['x'] = x
clustered_df['y'] = y
clustered_df['category'] = category
clustered_df


# In[21]:


sns.scatterplot(data = clustered_df, x = 'x', y = 'y', hue = 'category')


# In[ ]:




