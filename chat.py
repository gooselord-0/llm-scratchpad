from os import system
from pathlib import Path
from time import time
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  Conversation,
  pipeline
)

def list_model_directories(base_path):
    return [d.name for d in Path(base_path).iterdir() if d.is_dir()]

def user_select_model(models):
    system('clear')
    print("Available models:\n")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")
    selection = input("\nSelect a model by number: ")
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

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=False,
                                             revision="main")


# Need a better solution for this
if (selected_model_name == 'distilgpt2'):
  model.to('cpu')
elif ('bnb' not in selected_model_name):
  model.to('cuda')

# Change me
system_prompt = "You are a chatbot who responds to every question with an analogy about kangaroos"

# Chat template & system prompt
chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|> ' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
messages = [
  {
    "role": "system",
    "content": system_prompt,
  },
  {
    "role": "user", 
    "content": "I'd like to ask you a question."
  },
 ]

tokenizer = AutoTokenizer.from_pretrained(
  model_path,
  add_generation_prompt=True,
  chat_template=chat_template,
  truncation_size='Left',
  use_fast=True
)

chatbot = pipeline(
  "conversational",
  do_sample=True,
  eos_token_id=tokenizer.encode("<|im_end|>"),
  max_new_tokens=150,
  model=model,
  pad_token_id=tokenizer.eos_token_id,
  # (defaults to 1.0) - The value used to module the next token probabilities that will be used by default in the generate method of the model. Must be strictly positive.
  temperature=0.7,
  tokenizer=tokenizer,
  # (defaults to 50) — Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
  top_k=10,
  # (defaults to 1) — Value that will be used by default in the generate method of the model for top_p. If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
  top_p=0.05,
  # (defaults to 1) — Parameter for repetition penalty that will be used by default in the generate method of the model. 1.0 means no penalty.  
  repetition_penalty=1.1
)

system('clear')
print("\nModel >> Speak, human.")

while True:
  user_input = input("\nHuman >> ")
  conversation = Conversation(messages)

  conversation.add_message({
       "role": "user", 
       "content": user_input
    },
 )

  start_time = time()
  response = chatbot(conversation)
  end_time = time()
  elapsed_time = round(end_time - start_time, 2)

  print(f"\nModel >> [{elapsed_time}sec] {response.generated_responses[-1]}")
  