import pandas as pd
from transformers import AutoTokenizer
import re
from utils import format_input

# Read the CSV file
df = pd.read_csv('data/Acad.csv')
df['Instructor Answers'] = df['Instructor Answers'].replace('No instructor answer', '')
df['Student Answers'] = df['Student Answers'].replace('No student answer', '')

# Create the formatted input string combining subject and question
df['input'] = df.apply(lambda x: format_input(x["Subject"], x["Question"]), axis=1)

# Create the formatted output string combining instructor and student answers
df['output'] = df.apply(lambda x: x['Instructor Answers'] if x['Instructor Answers'] else "", axis=1)

# Select only the required columns
df = df[['input', 'output']]

df = df.dropna(subset=['input', 'output'])

# Clean HTML and special characters
def clean_text(text):
    # Replace HTML entities
    text = text.replace('&#39;', "'")
    text = text.replace('&#34;', '"')
    text = text.replace("#pin", "")
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,\"\'\?]|@(?=[^\d])', '', text)
    return text

df['input'] = df['input'].apply(clean_text)
df['output'] = df['output'].apply(clean_text)

# Remove rows with empty/None values
df = df[df['input'].str.strip() != '']
df = df[df['output'].str.strip() != '']

# Load tokenizer and filter by token length
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
input_lengths = [len(tokenizer.encode(str(text))) for text in df['input']]
output_lengths = [len(tokenizer.encode(str(text))) for text in df['output']]
df['input_length'] = input_lengths
df['output_length'] = output_lengths

# Filter rows where both input and output are <= 256 tokens
#df = df[df['input_length'] <= 256]
#df = df[df['output_length'] <= 256]
df = df[['input', 'output']]

# Save as CSV in format compatible with qlora dataloader
df.to_csv('data/acad_formatted.csv', index=False)

print("Data wrangling complete. Saved to acad_formatted.csv")
print(f"Dataset contains {len(df)} examples")
print("\nSample entry:")
print(f"\nInput:\n{df['input'].iloc[0]}")
print(f"\nOutput:\n{df['output'].iloc[0]}")
