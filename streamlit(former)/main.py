import streamlit as st
import pandas as pd
import numpy as np
import os
from zhipuai import ZhipuAI
API_KEY = os.environ.get('ZHIPU_KEY')

st.set_page_config(
    page_title="GLM4 API Demo"
)
st.title("GLM4 API Demo")

with st.sidebar:
    top_p = st.slider("top_p", 0.0, 1.0, 0.7,step = 0.01)
    temperature = st.slider("temperature", 0.0, 1.0, 0.8,step = 0.01)
    clear = st.button("清空历史对话")

def stream_chat(model_name,messages,top_p=0.7,temperature=0.8):
    client = ZhipuAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=top_p,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
    )
    return response

prompt_text = st.chat_input(
    '请输入：',
    key = 'prompt'
)

if clear:
    prompt_text = ""

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if clear:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_text:
    message = [{"role": "user", "content": prompt_text}]
    st.session_state.chat_history.append(message[0])
    st.chat_message(message[0]["role"]).write(message[0]["content"])
    placeholder = st.empty()
    response = stream_chat("glm-4",st.session_state.chat_history,top_p,temperature)
    message[0] = {"role": "assistant", "content": ""}
    for trunk in response:
        message[0]["content"] += trunk.choices[0].delta.content
        placeholder.chat_message(message[0]["role"]).write(message[0]["content"])
    st.session_state.chat_history.append(message[0])