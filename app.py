# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:34:00 2023

@author: 39333
"""

import pandas as pd
import streamlit as st
import langchain
import openai
import os
from streamlit_chat import message
openai.api_key=st.secrets['OPENAI_API_KEY']
#os.environ["OPENAI_API_KEY"] = '{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}'
#from langchain.llms import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
#from langchain.llms import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
#openai.api_key=st.secrets['OPEN_APY_KEY']
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        response = agent.run(prompt)

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)


st.title('chat-advisor]☕')#(':blue[ambrogio\'s Data Analysis Chatbot] ☕')
uploaded_file = st.file_uploader("Choose a csv file", type='csv')

if uploaded_file is not None:

    csv_data = uploaded_file.read()
    with open(uploaded_file.name, 'wb') as f: 
        f.write(csv_data)

    df = pd.read_csv(uploaded_file.name)
    st.dataframe(df.head(10))

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    agent = create_pandas_dataframe_agent(chat, df, verbose=True)

    st.text_input("Ask Something:", key="user")
    st.button("Send", on_click=send_click)

    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            message(st.session_state.responses[i], key=str(i), seed='Milo')
            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)
