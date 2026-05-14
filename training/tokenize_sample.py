from transformers import AutoTokenizer

# Load mBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

# Sample multilingual text
sample_text = """
भारत ने मैच जीत लिया।
India won the match.
ভারত ম্যাচ জিতেছে।
"""

# Tokenize sample text
encoding = tokenizer(
    sample_text,

    padding="max_length",

    truncation=True,

    max_length=32,

    return_tensors="pt"
)

# Show input IDs
print("Input IDs:")
print(encoding["input_ids"])

print("\n")

# Show attention mask
print("Attention Mask:")
print(encoding["attention_mask"])

print("\n")

# Convert token IDs back into readable tokens
tokens = tokenizer.convert_ids_to_tokens(
    encoding["input_ids"][0]
)

print("Tokens:")
print(tokens)