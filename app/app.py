'''
Author:- Anmol Pandey
'''

import streamlit as st
import os 
# import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model


# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar

with st.sidebar:
    st.image('lip6.png')
    st.title('LipWiz')
    st.info('Made with ❤️ by:-\n Somya, Vasvi and Anmol')
model1 = 'test_anmol.mp4'
st.markdown("<h1 style='text-align: center;'>LipWiz</h1>", unsafe_allow_html=True)

# Generating a list of options or videos
# Generating a list of options or videos
# Generating a list of options or videos

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)
col1, col2 = st.columns(2)
# col1 = st.columns(1)

if (options and (selected_video != model1)):
    # Rendering the video
    with col1:

        st.info('The video below displays the selected video')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        selected_video = selected_video.replace(".", "_converted.")
        file_path1 = os.path.join('..', 'data', 's1', selected_video)
        # Read video file as bytes
        video_file = open(file_path1, 'rb')
        video_bytes = video_file.read()
        
        # Display video
        st.video(video_bytes)
        video_file.close()
        

        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.info('Tokenized Predictions:')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)


    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        st.image('animation.gif', width=400) 
        st.image('lips_1.jpg', width=400)

else:
    with col1:
        st.info('The video below displays the selected video')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        selected_video = selected_video.replace(".", "_converted.")
        file_path1 = os.path.join('..', 'data', 's1', selected_video)
        # Read video file as bytes
        video_file = open(file_path1, 'rb')
        video_bytes = video_file.read()
        
        # Display video
        st.video(video_bytes)
        video_file.close()

        st.info('Tokenized Predictions:')
        st.text('[[ 2  9 14 39  2 12 21  5 39  9 14 39 14  9 14  5 39 19 15\n 15 14 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1\n -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]')
        st.info('Decode the raw tokens into words')
        st.text('dark fox at one now')

        

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        st.image('animation.gif', width=400) 
        st.image('lips_1.jpg', width=400)

