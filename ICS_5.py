#!/usr/bin/env python
# coding: utf-8

# In[132]:


import math
import random


# In[137]:


P = 11


# In[141]:


def modmul(a,b,m=P):
    return ((a%m)*(b%m))%m

def mod_pow(a,b,m=P):
    if b == 0:
        return 1
    r = mod_pow(a,b//2,m)
    r = (r*r)%m
    if b%2:
        r = (r*a)%m
    return r

def moddiv(a,b,m=P):
    return modmul(a,mod_pow(b,m-2,m),m)


# In[134]:


class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y


# In[152]:


class EllipticCurve:
    def __init__(self,a,b):
        self.a = a
        self.b = b
        
    def add(self,p1,p2,m=P):
        l = 0
        if p1 == p2:
            num = 3*p1.x*p1.x + self.a
            den = 2*p1.y
        else:
            num = p2.y - p1.y
            den = p2.x - p1.x
        
        l = moddiv(num,den,m)
        x3 = (l*l - p1.x - p2.x)%m
        y3 = (l*(p1.x-x3) - p1.y)%m
        
        return Point(x3,y3)
    
    def mul(self,k,p):
        temp = p
        while k !=1:
            temp = self.add(temp,p)
            k -= 1
        return temp
    
    def sub(self,p1,p2):
        np = Point(p2.x,-p2.y)
        return self.add(p1,np)


# In[153]:


curve = EllipticCurve(2,4)
G = Point(0,2)


# In[154]:


plainText = Point(3,4)


# In[155]:


Pv = 5
Pb = curve.mul(Pv,G)


# In[159]:


def encrypt(p,Pb):
    k = 5
    c = [
        curve.mul(k,G),
        curve.add(p,curve.mul(k,Pb))
    ]
    return c


# In[160]:


def decrypt(c,Pv):
    return curve.sub(c[1],curve.mul(Pv,c[0]))


# In[161]:


ciphertext = encrypt(plainText, Pb)
p = decrypt(ciphertext, Pv)


# In[163]:


print(p.x,p.y)


# In[ ]:




