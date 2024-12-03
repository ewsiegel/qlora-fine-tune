from utils import load_model
from utils import format_input

def generate_text(subject, question, model, tokenizer, max_new_tokens=256):
    formatted_prompt = format_input(subject, question)
    # Format the input with subject and question structure
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

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