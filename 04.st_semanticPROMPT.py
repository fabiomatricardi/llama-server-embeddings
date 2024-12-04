# Let's test a semantic evaluator to force a SML to say I DON'T KNOW
# here using pure llama.cpp server with llama-server.exe -m .\model\tiny-vicuna-1b.q6_k.gguf -c 2048 --port 8001
import streamlit as st
# mTinyVicuna1Bain: server is listening on http://127.0.0.1:8001 
# embeddings: server is listening on http://127.0.0.1:8002 
#
# You can use the following commands - they will open a server instance in 2 different windows
# start .\llama.cpp\llama-server.exe --embeddings -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\gte-small_fp16.gguf -c 1024 --port 8002
# start .\llama.cpp\llama-server.exe --embeddings -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\tiny-vicuna-1b.q6_k.gguf -c 2048 --port 8001
# PAUSE
#

from openai import OpenAI
import datetime
import random
import string
import configparser
import tiktoken
from sentence_transformers.util import pytorch_cos_sim, cos_sim, dot_score
from time import sleep

def countTokens(text):
    """
    Use tiktoken to count the number of tokens
    text -> str input
    Return -> int number of tokens counted
    """
    encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

# GOBALS
modelname = 'tiny-vicuna-1b.q6_k.gguf'
embeddingname = 'gte-small_fp16.gguf'

if "firstrun" not in st.session_state:
    st.session_state.firstrun = 0

standard = """One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward."""
#rewriting AVATARS  👷🐦  🥶🌀
av_us = 'user.png'  #or "🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = 'assistant.png'

# Set the webpage title
st.set_page_config(
    page_title=f"PROMPT AND RESPONSE EVALUATOR using LLM:{modelname} adn Embeddings:{embeddingname}",
    page_icon="🟠",
    layout="wide")

st.title("Validate question relevance in a prompt with context")

# ref https://docs.streamlit.io/develop/api-reference/caching-and-state
@st.cache_resource
def create_LLM():
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed", organization='SelectedModel')
    return client

@st.cache_resource
def create_EMB_MODEL():
    client = OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed")
    return client

def embed_QUERY(question,context,client):
    print("Create embeddings for 2 different texts and than compute similarity serach")
    print("Calling API endpoint for the 2 embeddings...")
    messages = [question,context]
    embedded = client.embeddings.create(
                            model="",
                            input=messages,
                            encoding_format="float"
                         )
    hits = pytorch_cos_sim(embedded.data[0].embedding, embedded.data[1].embedding)
    return hits


llm = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed", organization='SelectedModel')
embeddings = OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed")

with st.container():
    col1, col2 =  st.columns([0.6,0.4])
    with col1:
        st.subheader("Cosine similarity to declare *UNANSWERABLE* a question in RAG")
        st.markdown("""If the prompt is about summarization or listing topics the instruction will go through
            
Otherwise the embeddings with treshold 0.77 will cut off the user request and give automatic reply
            

- we can still refine the relevance of the reply with cosine similarity score, as a confidence indicator

""")
        
    with col2:
        with st.container(border=True):
            status = st.markdown('STATUS MESSAGES:   none')
        with st.container(border=True):
            results = st.markdown(f'Similarity score: ')
        rag_required = st.toggle(label='Required RAG')

if st.session_state.firstrun == 0:
    with st.spinner('working on it...'):
        status.markdown('STATUS MESSAGES:   LOADING MODEL API...')
        llm = create_LLM()
        st.toast('LLM API endopoint connected!', icon='🎉')
        sleep(2)
        status.markdown('STATUS MESSAGES:   done')
        sleep(1)
        status.markdown('STATUS MESSAGES:   LOADING MODEL API...')
        embeddings = create_EMB_MODEL()
        st.toast('Embeddings API endopoint connected!', icon='🎉')  
        sleep(2)
        status.markdown('STATUS MESSAGES:   done')
        sleep(1)
        st.session_state.firstrun =1

with st.container():
    c1, c2 = st.columns([0.5,0.5])
    with c1:
        user_input = st.text_area(label='Prompt:', height=90)
        user_context = st.text_area(label='Context:', height=200,disabled=not(rag_required))
        btn_inference = st.button(label='AI reply', type='primary')
    with c2:
        with st.container(border=True):
            reply = st.empty()    
if btn_inference:
        status.markdown('STATUS MESSAGES:   checking the query relevance with the context...') 
        sleep(2)
        status.markdown('STATUS MESSAGES:   relevance Score Cosine 0.7760...')
        sleep(1)
        status.markdown('STATUS MESSAGES:   Query and context relevant Question is answerable...')
        sleep(1)
        res = embed_QUERY(user_input,user_context,embeddings)
        st.toast("Let's do it", icon='😍') 
        sleep(1)
        results.markdown(f'Similarity score: **{res[0][0]:.5f}**')
        reply.markdown(standard)
        #results.write(res.data[1].embedding)


st.write('\n\n\n\n')
st.divider()
st.markdown("""Other possible similarity scores:
For some reasons are giving back the same numners...
```            
# compute similarity (3 methods)
from sentence_transformers.util import pytorch_cos_sim, cos_sim, dot_score
print('1 > pytorch_cos_sim...')
hits = pytorch_cos_sim(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(hits)
print('2 > cos_sim...')
hits = cos_sim(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(hits)
print('3 > dot_score...')
hits = dot_score(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(hits)
```
                        
### Additional notes
```
# a good threshold above 0.77  AnneFrank scores with China text 0.7258
# pollution question scores 0.7822  China and India scores 0.7980
# artificial intelligence scores 0.7488, what is science scores 0.7382
# advantages of chinese system 0.8242   
```
- @st.cache_resource does not work with OpenAI client instantiation
- @st.cache_resource cannot call another function under st.cache (for ecample `embed_QUERY(question,context,client)`)                               
""")
