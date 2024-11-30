from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
df = pd.read_csv('data/acad_formatted.csv')

# Calculate lengths
input_lengths = [len(tokenizer.encode(str(text))) for text in df['input']]
output_lengths = [len(tokenizer.encode(str(text))) for text in df['output']]

# Statistics
print("\nInput Text Statistics:")
print(f"Max length: {max(input_lengths)}")
print(f"95th percentile: {np.percentile(input_lengths, 95)}")
print(f"Mean length: {np.mean(input_lengths):.1f}")
print(f"Median length: {np.median(input_lengths)}")

print("\nOutput Text Statistics:")
print(f"Max length: {max(output_lengths)}")
print(f"95th percentile: {np.percentile(output_lengths, 95)}")
print(f"Mean length: {np.mean(output_lengths):.1f}")
print(f"Median length: {np.median(output_lengths)}")

# Plot distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(input_lengths, bins=50)
plt.title('Input Lengths Distribution')
plt.xlabel('Token Length')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(output_lengths, bins=50)
plt.title('Output Lengths Distribution')
plt.xlabel('Token Length')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('token_lengths.png')

# Calculate coverage at different lengths
def print_coverage(lengths, name):
    for length in [128, 256, 384, 512, 768, 1024]:
        coverage = sum(l <= length for l in lengths) / len(lengths) * 100
        print(f"{name} coverage at {length} tokens: {coverage:.1f}%")
    print()

print("\nCoverage Analysis:")
print_coverage(input_lengths, "Input")
print_coverage(output_lengths, "Output")