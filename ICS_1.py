#!/usr/bin/env python
# coding: utf-8

# In[42]:


#permutations for keys
p10_seq = (3, 5, 2, 7, 4, 10, 1, 9, 8, 6)
p8_seq = (6, 3, 7, 4, 8, 5, 10, 9)

#permutations for text
ip_seq = (2, 6, 3, 1, 4, 8, 5, 7)
inv_ip_seq = (4, 1, 3, 5, 7, 2, 8, 6)

#permutation to expand 4 bit to 8 bit
ep_seq = (4, 1, 2, 3, 2, 3, 4, 1)

#permutation for 4 bits
p4_seq = (2, 4, 3, 1)

s0_seq = [
            ["01", "00", "11", "10"],
            ["11", "10", "01", "00"],
            ["00", "10", "01", "11"],
            ["11", "01", "11", "10"]
         ]

s1_seq = [
            ["00", "01", "10", "11"],
            ["10", "00", "01", "11"],
            ["11", "00", "01", "00"],
            ["10", "01", "00", "11"]
         ]


# In[52]:


def get_permutation(inp,seq):
    s = ''
    for i in seq:
        s += inp[i-1]
    return s


# In[44]:


def find_xor(s1,s2):
    xor = ''
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            xor += '0'
        else:
            xor += '1'
    return xor


# In[45]:


def find_s0_s1(half,lookup_table):
    r = int(half[0])**2 + int(half[3])
    c = int(half[1])**2 + int(half[2])
    return lookup_table[r][c]


# In[46]:


def left_shift(ip,shift):
    return ip[shift:] + ip[:shift]


# In[54]:


def generate_keys(key):
    initial_per = get_permutation(key,p10_seq)
    
    left_half = initial_per[:5]
    right_half = initial_per[5:]
    
    ls1_left = get_shift(left_half,1)
    ls1_right = get_shift(right_half,1)
    
    k1 = get_permutation(ls1_left+ls1_right,p8_seq)
    
    ls2_left = get_shift(ls1_left,2)
    ls2_right = get_shift(ls1_left,2)
    
    k2 = get_permutation(ls2_left+ls2_right,p8_seq)
    
    return k1,k2


# In[48]:


def encrypt_round(left,right,key):
    expanded_per = get_permutation(right,ep_seq)
    expanded_per_xor = find_xor(expanded_per,key)
    
    left_half = expanded_per_xor[:4]
    right_half = expanded_per_xor[4:]
    
    s0 = find_s0_s1(left_half,s0_seq)
    s1 = find_s0_s1(right_half,s1_seq)
    
    p4 = get_permutation(s0+s1,p4_seq)
    
    left = find_xor(left,p4)
    
    return left,right


# In[49]:


def encrypt(inp,k1,k2):
    initial_per = get_permutation(inp,ip_seq)
    
    left,right = initial_per[:4],initial_per[4:]
    
    left,right = encrypt_round(left,right,k1)
    
    left,right = right,left
    
    left,right = encrypt_round(left,right,k2)
    
    inv_per = get_permutation(left+right,inv_ip_seq)
    
    return inv_per


# In[56]:


k1,k2 = generate_keys("0010010111")
print(k1,k2)


# In[57]:


plaintext = "00101000"
ciphertext = encrypt(plaintext,k1,k2)

print("ciphertext : ", ciphertext)


# In[58]:


deciphered_text = encrypt(ciphertext,k2,k1)
print('deciphered_text : ', deciphered_text)


# In[ ]:




