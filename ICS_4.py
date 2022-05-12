#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import math


# In[10]:


def isPrime(n):
    if n == 0 or n == 1:
        return False
    
    for i in range(2,int(n**0.5)+1):
        if n%i == 0:
            return False
    return True

def generate_primes():
    primes = [i for i in range(0,999) if isPrime(i)]
    return random.choices(primes,k=2)


# In[21]:


class RSA:
    def __init__(self,p,q):
        self.p = p
        self.q = q
        self.n = p*q
        self.phi = (p-1)*(q-1)
        self.generate_keys()
       
    def generate_keys(self):
        for i in range(2,self.phi):
            if (math.gcd(self.phi,i)==1):
                self.e = i
                break
        
        for i in range(2,99999):
            if (i*self.phi+1)%self.e ==0:
                self.d = int((i*self.phi+1)/self.e)
                break
    
    def encrypt(self,text):
        pt = []
        ct = []
        
        for i in text:
            pt.append(ord(i))
        
        for i in pt:
            ct.append((i**self.e)%self.n)
        
        return ct
            
    def decrypt(self,ct):
        
        dt = []
        for i in ct:
            dt.append(chr((i**self.d)%self.n))
            
        return "".join(dt)
        


# In[23]:


if __name__ == "__main__":
    
    p,q = generate_primes()
    print('Generated Prime nos are p = {} , q = {}'.format(p,q))
    rsa = RSA(p,q)
    
    
    plain_text = str(input('Enter the plain text: '))
    
    ct = rsa.encrypt(plain_text)
    
    print('Encrypted text : {}'.format(ct))

    decrypted_text = rsa.decrypt(ct)

    print('Descrypted Message : {}'.format(decrypted_text))


# In[ ]:





# In[ ]:




