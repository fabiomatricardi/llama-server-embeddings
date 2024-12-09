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
import configparser

def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('tiny-vicuna-1b.ini',encoding='utf-8')
    # Access values from the configuration file
    NCTX = config.getint('Model', 'NCTX')
    modelname = config.get('Model', 'name')
    modelfile = config.get('Model', 'file')
    STOPS = config.get('Model', 'STOPS')
    myheader = config.get('UI', 'myheader')
    cursor = config.get('UI', 'cursor')
    av_us = config.get('UI', 'av_us')
    av_ass = config.get('UI', 'av_ass')
    # Return a dictionary with the retrieved value
    return NCTX,modelname,modelfile,STOPS,myheader,cursor,av_us,av_ass

NCTX,modelname,modelfile,STOPS,myheader,cursor,av_us,av_ass = read_config()

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

standard1 = """One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward."""
standard = """Large language models (LLMs) have shown great capabilities but also have flaws. These include the vice of producing hallucinations or the presence of outdated content. This has led to the emergence of a new paradigm called retrieval augmented generation (RAG). Previous LLMs had a limited context length (usually no more than 4096), which significantly limited the context that could be entered into the prompt. This meant time-consuming and laborious optimization work to find the appropriate context. In fact, one of the sore points of RAG is chunking and the need to choose a suitable chunking strategy for one’s data. Over the years intensive research has been devoted to extending the context length of today’s LLMs precisely to reduce this problem and be able to provide more context to the model."""
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
        st.markdown(f"### TinyVicuna reply")
        with st.container(border=True):
            reply = st.empty()    
        st.markdown(f"###### *About similarity score threshold*")
        with st.popover("Additional info about simScore"):
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
if btn_inference:
        if rag_required:
            sleep(1)
            st.toast("Let's do it", icon='😍') 
            status.markdown('STATUS MESSAGES:   checking the query relevance with the context...') 
            res = embed_QUERY(user_input,user_context,embeddings)
            simscore = float(res[0][0])
            if simscore>=0.7760:
                sleep(2)
                status.markdown('STATUS MESSAGES:   relevance Score Cosine threshold >= 0.7760...')
                sleep(1)
                status.markdown('STATUS MESSAGES:   Query and context relevant Question is **:green[answerable...]**')
                results.markdown(f'Similarity score: **:green[{simscore:.5f}]**')
                sleep(1)
                status.markdown('STATUS MESSAGES:   **:green[Generating reply...]**')
                prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Reply to the question only using the provided context.
[context]{user_context}[end of context]

Question: {user_input}
ASSISTANT:"""
                full_response = ''
                for items in llm.completions.create(
                                prompt =prompt,
                                model=modelname,
                                temperature=0.15,
                                presence_penalty=1.35,
                                stop=STOPS,
                                max_tokens=500,              
                                stream=True):
                    full_response += items.content
                    reply.markdown(full_response + cursor)
                reply.markdown(full_response)                 
            else:
                sleep(2)
                full_response = ''
                status.markdown('STATUS MESSAGES:   relevance Score Cosine threshold >= 0.7760...')
                sleep(1)
                status.markdown('STATUS MESSAGES:   Query and context relevant Question is **:red[UNANSWERABLE...]**')            
                results.markdown(f'Similarity score: **:red[{simscore:.5f}]**')
                mymessage = "I DON'T KNOW. Based on the given context the question is UNANSWERABLE"
                for items in mymessage:
                    full_response += items
                    reply.markdown(full_response + cursor)
                    sleep(0.02)
                reply.markdown(full_response)
                #results.write(res.data[1].embedding)
        else:
            sleep(1)
            st.toast("Let's do it", icon='😍')             
            status.markdown('STATUS MESSAGES:   **:green[Generating reply...]**')
            prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Reply to the question only using the provided context.
[context]{user_context}[end of context]

Question: {user_input}
ASSISTANT:"""
            full_response = ''
            for items in llm.completions.create(
                            prompt =prompt,
                            model=modelname,
                            temperature=0.15,
                            presence_penalty=1.35,
                            stop=STOPS,
                            max_tokens=500,              
                            stream=True):
                full_response += items.content
                reply.markdown(full_response + cursor)
            reply.markdown(full_response)   
            
st.write('\n\n\n\n')
st.divider()

