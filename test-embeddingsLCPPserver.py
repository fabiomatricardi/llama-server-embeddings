# llama-server.exe --embeddings -m C:\Users\FabioMatricardi\Documents\DEV\LLAMACPP-GG\llama.cpp\models\e5-small-v2.Q8_0.gguf -c 512 --port 8002

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
#print(message1_em.data[0].embedding)
#print(message2_em.data[0].embedding)
#print('---')
print('Computing similarity...')
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