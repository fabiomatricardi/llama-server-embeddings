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

