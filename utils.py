import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftConfig

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

def load_model(
    base_model_name="huggyllama/llama-13b",
    adapter_path="ewsiegel/fine-tuned-llama-13b-acad",
    update_vocab_size=True,
    use_adapter=True
):
    if use_adapter:
        # Load and print adapter config from HF Hub
        config = PeftConfig.from_pretrained(adapter_path)
        print("Adapter config:", config)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="right",
        add_eos_token=False,
        add_bos_token=True,
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
    
    if use_adapter:
        print("Loading adapter from Hugging Face Hub...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        print("Merging model weights...")
        model = model.merge_and_unload()
    else:
        model = base_model
    
    return model, tokenizer