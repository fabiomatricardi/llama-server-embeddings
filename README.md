<img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/truthGPT_example.png' width=800>

# llama-server-embeddings
##### How to use llama-server for embeddings and similarity score
This app intent is to use embeddings and similarity score to FORCE the SML to say "I DON'T KNOW'

The trick here is to take a threshold in the cosine similarity: whatever is below the set-point is too far from the context, so it cannot be answered; everything else is treated as a normal RAG.

## what you need
- Download binareies of llama.cpp directly from the pre-compiled releases, according to your architecture
- extract the `.zip` file in the sub-folder `llama.cpp` (if you cloned the repo you will already find it there)
- Download your GGUF files (LLM or embeddings)in the sub-folder `llama.cpp\models` (if you cloned the repo you will already find it there)

> for my tests I used `e5-small-v2_fp16.gguf` as embeddings model
>
> you can download it from [here](https://huggingface.co/ChristianAzinn/e5-small-v2-gguf/resolve/main/e5-small-v2_fp16.gguf)

##### OpenAI API specifications
Check them [HERE](https://platform.openai.com/docs/api-reference/embeddings/object)

##### A story of Llama.cpp
[An amazing blog-journal](https://steelph0enix.github.io/posts/llama-cpp-guide/)

## Dependencies
```
pip install openai streamlit tiktoken sentence-transformers pillow
```

---

> 2024-12-03



## with Streamlit
create a batch file with the following lines (something like `runservers.bat`)

In my case the models are in the subdirectory `D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model`

```
start .\llama.cpp\llama-server.exe --embeddings -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\e5-small-v2_fp16.gguf -c 1024 --port 8002
start .\llama.cpp\llama-server.exe -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\tiny-vicuna-1b.q6_k.gguf -c 2048 --port 8001
PAUSE
```
- this will open a server instance in 2 different windows
<img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/tri-windows.png' width=900>

- clone the repo
- run from the temrinal with the venv activated:
```
streamlit run .\04.st_semanticPROMPT.py
```
>  You can now view your Streamlit app in your browser.
>
>  Local URL: http://localhost:8501
>  Network URL: http://172.16.19.83:8501

<img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/20241209-semBAD.png' width=400><img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/20241209-semOK.png' width=400>

#### Work in progress
For now I am only computing the similarity.

Next steps:
- check if the prompt is about summarization, list of topics
- [x] check if RAG is required (there is a toggle)
- [x] if no RAG required, LLM is called for the reply
- [x] if RAG is required,
  - [ ] check if the context or query is empty
  - [x] compute similarity score
  - [x] if cosine similarity >= 0.77 call the LLM and
  - [ ] compute confidence score between prompt and reply
  - [x] ELSE throw a message (UNANSWERABLE, I DON'T KNOW, MISSING data)

### Details
two endpoints on different PORTS are running so we need to client connections:
```python
llm = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed", organization='SelectedModel')
embeddings = OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed")
```
Cosine Similarity is computed with `sentence-transformers` and the embedded texts (query,context) are passed to the function:
```python
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
    simscore = float(hits[0][0])
    return simscore
```
The function compute first  a list with two embedded documents, and then calculate the cosine similarity

the retrned value is a tensor type so later we need to extract it
```python
results.markdown(f'Similarity score: **{res:.5f}**')
```

<img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/workflow.png' width=800>



#### Footnotes on Streamlit
Other possible similarity scores:
For some reasons are giving back the same numbers...
```python         
# compute similarity (3 methods)
from sentence_transformers.util import pytorch_cos_sim, cos_sim, dot_score
print('1 > pytorch_cos_sim...')
hits = pytorch_cos_sim(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(float(hits[0][0]))
print('2 > cos_sim...')
hits = cos_sim(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(float(hits[0][0]))
print('3 > dot_score...')
hits = dot_score(message1_em.data[0].embedding, message2_em.data[0].embedding)
print(float(hits[0][0]))
```
                        
#### Additional notes
```
# a good threshold above 0.77  AnneFrank scores with China text 0.7258
# pollution question scores 0.7822  China and India scores 0.7980
# artificial intelligence scores 0.7488, what is science scores 0.7382
# advantages of chinese system 0.8242   
```
- `@st.cache_resource` does not work with OpenAI client instantiation
- `@st.cache_resource` cannot call another function under st.cache (for example `embed_QUERY(question,context,client)`)


---
> 2024-12-03

## Usage
From the main project directory
- in one terminal window run
```
.\llama.cpp\llama-server.exe --embeddings -m C:\Users\FabioMatricardi\Documents\DEV\LLAMACPP-GG\llama.cpp\models\e5-small-v2.Q8_0.gguf -c 512 --port 8002
```


- in another terminal window run
```python
from openai import OpenAI
from sentence_transformers.util import pytorch_cos_sim
import sys

# instance for API call to the embeddings endpoint
embeddings = OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed")
THRESHOLD = 0.7760

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
    simscore = float(hits[0][0])
    return simscore

while True:
    #start a while loop here
    userinput1 = ""
    userinput2 = ""
    print("\033[1;30m")  #dark grey
    print("Enter your first text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[0m")  #reset all colors
    lines = sys.stdin.readlines()
    for line in lines:
        userinput1 += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break #for the while loop if exists
    print("\033[1;30m")  #dark grey
    print("Enter your second text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[0m")  #reset all colors
    lines2 = sys.stdin.readlines()
    for line in lines2:
        userinput2 += line + "\n"
    if "quit!" in lines2[0].lower():
        print("\033[0mBYE BYE!")
        break #for the while loop if exists

    print('---')
    print("Calling API endpoint for embeddings and Computing similarity...")
    relevance = embed_QUERY(userinput1,userinput2,embeddings)
    print('')
    if relevance >= THRESHOLD:
        print("\033[92;1m")  #green
        print(f'Relevance score = {relevance:.5f}\ncalling the Small Language Model for inference...')
        print("\033[0m")  #reset all colors
    else:
        print("\033[91;1m")  #red    
        print(f'UNANSWERABLE! \nRelevance score only {relevance:.5f}')
        print("\033[0m")  #reset all colors
    print('---')

# a good threshold above 0.77  AnneFrank scores with China text 0.7258
# pollution question scores 0.7822  China and India scores 0.7980
# artificial intelligence scores 0.7488, what is science scores 0.7382
# advantages of chinese system 0.8242
```

### Referenced text used for the tests
The main objective is to get even a small model to be able to say "I don't know"

Small Language Models are often unable to follow the instruction when they have a knowledge of the question.

If I used the mentione below context and ask the model "Who is Anne Frank" the model is going to reply....

But we don't want it. **Based on the context the question is UNANSWERABLE**
```
One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
```


#### Streamlit base tutorial for chatbots
tutorial [here](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming)


---

# rev 044 - using smollm2-360m-instruct-q8_0.gguf
Add 3 different embedding models
> noted that every model gives quite different cosine_similarity
> this is a problem to decide the threshold
> -
> gte-small is less inclusive than e5-small, but both are GOOD
> -
> all-MiniLM-V12 not easy


#### the batch file
```batch
echo e5-small-v2_fp16.gguf -c 1024 --port 8002
echo all-MiniLM-L12-v2.Q8_0.gguf -c 1024 --port 8003
echo gte-small_fp16.gguf -c 1024 --port 8004
echo model smollm2-360m-instruct-q8_0.gguf -c 2048 --port 8001

start .\llama.cpp\llama-server.exe --embeddings -m C:\Users\FabioMatricardi\Documents\DEV\PortableLLMS\TruthGPT\llama.cpp\models\e5-small-v2_fp16.gguf -c 1024 --port 8002

start .\llama.cpp\llama-server.exe --embeddings -m C:\Users\FabioMatricardi\Documents\DEV\PortableLLMS\TruthGPT\llama.cpp\models\all-MiniLM-L12-v2.Q8_0.gguf -c 1024 --port 8003

start .\llama.cpp\llama-server.exe --embeddings -m C:\Users\FabioMatricardi\Documents\DEV\PortableLLMS\TruthGPT\llama.cpp\models\gte-small_fp16.gguf -c 1024 --port 8004

start .\llama.cpp\llama-server.exe -m C:\Users\FabioMatricardi\Documents\DEV\PortableLLMS\TruthGPT\llama.cpp\models\smollm2-360m-instruct-q8_0.gguf -c 4096 --port 8001


PAUSE

```

### how to run
from the terminal, with `venv` activated run
```
streamlit run .\044.st_semanticPROMPT.py
```


<img src='https://github.com/fabiomatricardi/llama-server-embeddings/raw/main/ample-images/2024-12-11%2018%2037%2038.png' width=900>






