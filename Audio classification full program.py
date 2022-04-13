#!/usr/bin/env python
# coding: utf-8

# ### Setting the working directory

# In[1]:


import os
os.chdir('E:\\DBDAproject\\UrbanSound8K')
os.getcwd()


# ### Loading the audio file with Librosa and Scipy 
# Librosa: Librosa is a Python package for music and audio analysis, it is basically used when we work with audio data. It provides the building blocks necessary to create the music information retrieval systems.we have used Librosa to load audio data and plot the waveform. When we load any audio file with Librosa, it gives us 2 things. One is sample rate, and the other is a two-dimensional array. It normalizes the entire data and tries to give it in a single sample rate.
# Scipy: Scipy can be used to read and write a wav file. When you print the sample rate using scipy it is different than librosa. When we print the data retrieved from librosa, it can be normalized, but when we try to read an audio file using scipy, it can’t be normalized.

# In[2]:


import librosa
audio_file_path='7389-1-0-3.wav'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)


# In[3]:


print(librosa_audio_data)


# In[4]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)


# In[5]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(audio_file_path)


# In[6]:


wave_audio


# In[7]:


plt.figure(figsize=(12, 4))
plt.plot(wave_audio)


# ### Extracting features of the audio file using MFCC
# Mel-Frequency Cepstral Coefficients (MFCC) is used for feature extraction from the audio samples.MFCC algorithm summarizes the frequency distribution across the window size. This enables the analysis of both the frequency and time characteristics of the provided sound. It will allow us to identify features for classification.

# In[8]:


mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)


# In[9]:


from matplotlib import cm
fig, ax = plt.subplots(figsize=(15, 20))
cax = ax.imshow(mfccs, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC')

plt.show()


# In[10]:


mfccs


# ### Reading the csv file metadata.csv 
# meatdata.csv : This file contains meta-data information about every audio file in the dataset.

# In[11]:


import pandas as pd
import os
import librosa

audio_dataset_path='audio/'
metadata=pd.read_csv('metadata/UrbanSound8K.csv')
metadata.head()


# ### Generalizing the MFCC code for entire dataset 

# In[12]:


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


# In[13]:


import numpy as np
from tqdm import tqdm
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[14]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[15]:


extracted_features_df.to_csv('extracted_features.csv',index=False)


# ### Train test split

# In[16]:


X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[17]:


X.shape


# In[18]:


y


# In[19]:


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[20]:


y[0]


# In[21]:


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[22]:


X_train


# In[23]:


y


# In[24]:


X_train.shape,X_test.shape


# In[25]:


y_train.shape,y_test.shape


# ### Model building 

# In[26]:


import tensorflow as tf
print(tf.__version__)


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from sklearn import metrics


# In[28]:


# No of classes 10 
num_labels=y.shape[1]


# In[67]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)


# In[68]:


optimizer = Adam( learning_rate= lr_schedule)


# In[69]:


model=Sequential()
# First layer
model.add(Dense(500,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# Second layer
model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.1))

# Third layer
model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.1))

# Fourth layer
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# Final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[70]:


model.summary()


# ### Training the model 

# In[71]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer= optimizer)


# In[72]:


## Trianing the model
from tensorflow.keras.callbacks import ModelCheckpoint

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_model/audio_classification2.hdf5', 
                               verbose=1, save_best_only=True)

history=model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


# ### visualizing the result of model 
# 1. Loss curve 
# 2. Accuracy curve 

# In[73]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[74]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[75]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[76]:


X_test[1]


# In[77]:


model.predict(X_test)


# ### Testing the model with new audio file  

# In[78]:


import numpy as np
filename="28808-1-0-7.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(" 1 :",mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(" 2 :",mfccs_scaled_features)
print(" 3 :",mfccs_scaled_features.shape)
predicted_label=np.argmax(model.predict(mfccs_scaled_features), axis=-1)
print(" 4 :",np.array(predicted_label))
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class

