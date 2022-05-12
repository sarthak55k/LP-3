#!/usr/bin/env python
# coding: utf-8

# # Diffie-Hellman key exchange

# In[2]:


import numpy as np 
import random
import math


# In[3]:


def isPrime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
    return True

def isPrimitive(n,G):
    d = {}
    if G>=n:
        return False
    else:
        for x in range(n-1):
            z = pow(G,x,n)
            if z in d:
                return False
            d[z] = 1
    return True
        
        


# In[7]:


def generate(n,G):
    a = random.randint(2,1999)
    b = random.randint(2,1999)
    
    xa = pow(G,a,n)
    xb = pow(G,b,n)
    
    print("Generated xa: ",xa)
    print("Generated xb: ",xb)
    
    ka = pow(xb,a,n)
    kb = pow(xa,b,n)
    
    print("Generated ka: ",ka)
    print("Generated kb: ",kb)


# In[14]:


if __name__ == "__main__":
    n = int(input('Enter a prime no: '))
    while not isPrime(n):
        print("Not a prime no!")
        n = int(input('Enter a prime no: '))
    
    G = int(input("Enter the primitive root of {}\n".format(n)))
    while not isPrimitive(n,G):
        print("Not a primitive root!")
        G = int(input("Enter the primitive root of {}\n".format(n)))
    
    generate(n,G)


# In[ ]:





# In[ ]:




