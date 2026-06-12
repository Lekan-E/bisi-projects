import pandas as pd
import pickle
import streamlit as st

# Load pretrained models
lr_pickle = open('models/LRmodel.pkl', 'rb')
lr_model = pickle.load(lr_pickle)
lr_pickle.close()

rf_pickle = open('models/RFmodel.pkl', 'rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()