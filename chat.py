import time
from os import system
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, pipeline

def list_model_directories(base_path):
    return [d.name for d in Path(base_path).iterdir() if d.is_dir()]

def user_select_model(models):
    print("Available models:")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")
    selection = input("Select a model by number: ")
    try:
        selected_model = models[int(selection) - 1]
        return selected_model
    except (IndexError, ValueError):
        print("Invalid selection. Please select a valid number.")
        return user_select_model(models)

# Update as needed
model_dir = 'Repos/huggingface'

models_path = f'{Path.home()}/{model_dir}'

model_directories = list_model_directories(models_path)

# Let the user select a model
selected_model_name = user_select_model(model_directories)

# Construct the path to the selected model
model_path = f'{models_path}/{selected_model_name}/'

print(f"Selected model: {selected_model_name}")
print(f"Model path: {model_path}")

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=False,
                                             revision="main")

if (selected_model_name == 'distilgpt2') or (selected_model_name == 'gemma-2b-it'):
  model.to('cpu')
elif (selected_model_name != 'gemma-2b-it') and (selected_model_name != 'gemma-2b-it-bnb-4bit'):
  model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

chatbot = pipeline(
  "conversational",
  model=model,
  tokenizer=tokenizer,
  max_new_tokens=50,
  do_sample=True,
  pad_token_id=tokenizer.eos_token_id,
  # Higher temperature will make outputs more random and diverse
  temperature=0.05,
  # Lower top-p values reduce diversity and focus on more probable tokens
  top_p=0.05,
  # Lower top-k also concentrates sampling on the highest probability tokens for each step  
  top_k=10,
  repetition_penalty=1.1
)

system('clear')
print("\nModel >> Speak, human.")

while True:
  user_input = input("\nHuman >> ")
  conversation = Conversation(user_input)
  
  start_time = time.time()
  response = chatbot(conversation)
  end_time = time.time()
  elapsed_time = end_time - start_time

  print(f"\nModel >> [{elapsed_time}sec] {response.generated_responses[-1]}")
  