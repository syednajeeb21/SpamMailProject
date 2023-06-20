# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:54:33 2023

@author: SAYYID NAJEEB
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle

DeepLearningModel = keras.models.load_model("Spam_mail_ANN.h5")

st.title("Email Spam Detection")
st.image("https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2020/05/Untitled-46.png?resize=600%2C353&ssl=1")
model = pickle.load(open("Spam_mail_prediction", 'rb'))
train = pickle.load(open("MailData", 'rb'))
text = st.text_input('Enter your Email Here!')
tff = TF(stop_words='english', lowercase=True)
train_features = tff.fit_transform(train)
text = [text]
text = tff.transform(text)

matrixtext = text.toarray()

#ModelList = ['Deep Learning' 'Machine Learning']

x = st.selectbox('Select the Type of Model', ('Deep Learning', 'Machine Learning'))

if x == 'Deep Learning':
    if st.button('Predict'):
        pred = DeepLearningModel.predict(matrixtext)
        pred = np.argmax(pred)
        if pred == 1:
            st.success('Given Mail is Ham')
        else:
            st.success('Spam Mail')

else:
    if st.button('Predict'):
        pred = model.predict(text)
        if pred == 1:
            st.success('Given Mail is Ham Mail')
        else:
            st.success('Spam Mail')
