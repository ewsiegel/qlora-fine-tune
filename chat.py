from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os

base_model_name = "huggyllama/llama-13b"
adapter_path = "./output2/checkpoint-1050/"
update_vocab_size = True

def load_model():
    
    # First, let's check the adapter config
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
        print("Adapter config:", json.dumps(adapter_config, indent=2))
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="right",
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Only add special tokens and resize if update_vocab_size is True
    if update_vocab_size:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Vocabulary size after adding PAD: {len(tokenizer)}")
    
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load base model first with correct vocab size
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Conditionally resize token embeddings
    if update_vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
        print("Resized token embeddings to match tokenizer vocabulary size.")
    
    print("Loading adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print("Merging model weights...")
    model = model.merge_and_unload()
    
    return model, tokenizer

def format_input(subject, question):
    formatted_prompt = f"""
You are a helpful academic advisor assistant that provides answers to questions about general academic advising topics.

You will receive input in the following format:

subject: subject line,
question: student question

Your response should be in the following format:

answer: advisor answer

Here is the input:

subject: {subject},
question: {question}

Provide your helpful response here:
"""
    return formatted_prompt
def generate_text(subject, question, model, tokenizer, max_new_tokens=256):
    formatted_prompt = format_input(subject, question)
    # Format the input with subject and question structure
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response if it's included
    if response.startswith(formatted_prompt):
        response = response[len(formatted_prompt):].strip()
    
    return response

def interactive_chat():
    model, tokenizer = load_model()
    print("\nModel loaded! Enter your subject and question below. Type 'quit' at any time to exit.")
    print("-" * 50)
    
    while True:
        # Get subject
        subject = input("\nEnter subject: ").strip()
        if subject.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        # Get question
        question = input("Enter question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if subject and question:
            try:
                response = generate_text(subject, question, model, tokenizer)
                print("\nModel:", response.strip())
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        print("-" * 50)

if __name__ == "__main__":
    interactive_chat()