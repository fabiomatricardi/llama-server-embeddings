import streamlit as st
import random
import time


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            """### I DON'T KNOW.
            
The context does not provide enough information. 
**UNANSWERABLE** """,
        ]
    )
    for word in response:
        yield word
        time.sleep(0.05)

st.set_page_config(page_title='Liar GPT', page_icon='🫗', layout="wide")
av_as = '20241209-logoTruthGPT2.png'
av_us = 'user.png'
st.image('liarGPT.png', use_container_width=True)
#st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_as):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user",avatar=av_us):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant",avatar=av_as):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# "Given the following text, reply to the user question.\n[context]Large language models (LLMs) have shown great capabilities but also have flaws. These include the vice of producing hallucinations or the presence of outdated content. This has led to the emergence of a new paradigm called retrieval augmented generation (RAG). Previous LLMs had a limited context length (usually no more than 4096), which significantly limited the context that could be entered into the prompt. This meant time-consuming and laborious optimization work to find the appropriate context. In fact, one of the sore points of RAG is chunking and the need to choose a suitable chunking strategy for one’s data. Over the years intensive research has been devoted to extending the context length of today’s LLMs precisely to reduce this problem and be able to provide more context to the model. [end of context]\n Question: what is Science?"