#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('E:\\DBDA_project\\UrbanSound8K')
os.getcwd()


# In[2]:


import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


filename = '4201-3-0-0.wav'


# In[4]:


import librosa.display
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename )
librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[6]:


filename = 'Children_playing.wav'
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[7]:


sample_rate


# In[8]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio=wav.read(filename)


# In[9]:


wave_sample_rate


# In[10]:


wave_audio


# In[11]:


data


# In[12]:


import pandas as pd

metadata=pd.read_csv('metadata/UrbanSound8K.csv')
metadata.head(10)


# In[13]:


### Check whether the dataset is imbalanced
metadata['class'].value_counts()


# In[14]:


metadata.isnull().sum()

