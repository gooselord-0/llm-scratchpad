import time
from os import system
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, pipeline


# Update as needed
model_name = 'gemma-2b-it-bnb-4bit'
#model_name = 'Mistral-7B-Instruct-v0.2-GPTQ'
model_dir = 'Repos/huggingface'
model_path = f'{Path.home()}/{model_dir}/{model_name}'


model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")#.to('cuda')
                                            # Gemma 2B/7B incompatible with .to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

chatbot = pipeline(
  "conversational",
  attention_mask=None,
  model=model,
  tokenizer=tokenizer,
  max_new_tokens=150,
  do_sample=True,
  pad_token_id=tokenizer.eos_token_id,
  # Higher temperature will make outputs more random and diverse
  temperature=0.7,
  # Lower top-p values reduce diversity and focus on more probable tokens
  top_p=0.95,
  # Lower top-k also concentrates sampling on the highest probability tokens for each step  
  top_k=40,
  repetition_penalty=1.1
)

system('clear')
print("\nModel >> I sense your presence, mortal. Speak.")

while True:
  user_input = input("\nMortal >> ")
  conversation = Conversation(user_input)
  
  start_time = time.time()
  response = chatbot(conversation)
  end_time = time.time()
  elapsed_time = end_time - start_time

  print(f"\nModel >> [{elapsed_time}sec] {response.generated_responses[-1]}")
  