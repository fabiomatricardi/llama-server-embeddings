# llama-server-embeddings
How to use llama-server for embeddings and similarity score


## what you need
- Download binareies of llama.cpp directly from the pre-compiled releases, according to your architecture
- extract the `.zip` file in the sub-folder `llama.cpp` (if you cloned the repo you will already find it there)
- Download your GGUF files (LLM or embeddings)in the sub-folder `llama.cpp\models` (if you cloned the repo you will already find it there)

> for my tests I used `e5-small-v2.Q8_0.gguf` as embeddings model

## Dependencies
```
pip install openai streamlit tiktoken sentence-transformers pillow
```

---

> 2024-12-03

## with Streamlit
create a batch file with the following lines (something like `runservers.bat`
```
start .\llama.cpp\llama-server.exe --embeddings -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\gte-small_fp16.gguf -c 1024 --port 8002
start .\llama.cpp\llama-server.exe --embeddings -m D:\PortableLLMs\2024.TinyVicuna1B\llama.cpp\model\tiny-vicuna-1b.q6_k.gguf -c 2048 --port 8001
PAUSE
```
- this will open a server instance in 2 different windows

- clone the repo
- run from the temrinal with the venv activated:
```
streamlit run .\04.st_semanticPROMPT.py
```
>  You can now view your Streamlit app in your browser.
>
>  Local URL: http://localhost:8501
>  Network URL: http://172.16.19.83:8501

#### Work in progress
For now I am only computing the similarity.

Next steps:
- check if the prompt is about summarization, list of topics
- check if RAG is required (there is a toggle)
- if no RAG required, LLM is called for the reply
- if RAG is required,
  - check if the context or query is empty
  - compute similarity score
  - if cosine similarity >= 0.77 call the LLM and compute confidence score between prompt and reply
  - ELSE throw a message (UNANSWERABLE, I DON'T KNOW, MISSING data)

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
    return hits
```
The function compute first  a list with two embedded documents, and then calculate the cosine similarity

the retrned value is a tensor type so later we need to extract it
```python
results.markdown(f'Similarity score: **{res[0][0]:.5f}**')
```


#### Footnotes on Streanlit
Other possible similarity scores:
For some reasons are giving back the same numners...
```python         
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
                        
#### Additional notes
```
# a good threshold above 0.77  AnneFrank scores with China text 0.7258
# pollution question scores 0.7822  China and India scores 0.7980
# artificial intelligence scores 0.7488, what is science scores 0.7382
# advantages of chinese system 0.8242   
```
- `@st.cache_resource` does not work with OpenAI client instantiation
- `@st.cache_resource` cannot call another function under st.cache (for ecample `embed_QUERY(question,context,client)`)


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
import sys

client = OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed")
print("Create embeddings for 2 different texts and than compute similarity serach")
#start a while loop here
userinput1 = ""
userinput2 = ""
print("\033[1;30m")  #dark grey
print("Enter your first text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
print("\033[91;1m")  #red
lines = sys.stdin.readlines()
for line in lines:
    userinput1 += line + "\n"
if "quit!" in lines[0].lower():
    print("\033[0mBYE BYE!")
    #break #for the while loop if exists
print("\033[1;30m")  #dark grey
print("Enter your second text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
print("\033[91;1m")  #red
lines2 = sys.stdin.readlines()
for line in lines2:
    userinput2 += line + "\n"
if "quit!" in lines2[0].lower():
    print("\033[0mBYE BYE!")
    #break #for the while loop if exists
print("Calling API endpoint for the 2 embeddings...")
message1_em = client.embeddings.create(
                         model="",
                         input=userinput1,
                         encoding_format="float"
                         )
message2_em = client.embeddings.create(
                         model="",
                         input=userinput2,
                         encoding_format="float"
                         )
print('---')
print('Computing similarity...')

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


