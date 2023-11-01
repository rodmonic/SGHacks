import os
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
import re

# Path to add CSV file with full-text data
csv_file = 'data/arxiv_papers_full_v2_w_full_text.csv'

# Output CSV file of summaries
output_csv = 'summaries_full.csv'

# Initializing BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

summaries = []

# Regular expression pattern to remove special characters
special_chars_pattern = r'[^a-zA-Z\s]+'

# Read the input CSV file with full-text data
df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    full_text = row['Full Text']
    pdf_url = row['PDF URL']  # Get the PDF URL

    try:
        # Check if full_text exceeds the model's token limit
        if len(tokenizer.encode(full_text)) > model.config.max_position_embeddings:
            # The text is too long; truncate or split it
            # For simplicity, this code just truncates the text
            # You can implement more sophisticated logic for splitting
            full_text = full_text[:model.config.max_position_embeddings - 2]  # Leave space for special tokens

        # Tokenize the full text
        inputs = tokenizer(full_text, return_tensors='pt', max_length=1024, truncation=True)

        # Summary generation
        summary_ids = model.generate(inputs['input_ids'], num_beams=5, min_length=150, max_length=350, early_stopping=True)
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Remove special characters and normalize text
        summarized_text = re.sub(special_chars_pattern, ' ', summarized_text)
        summarized_text = ' '.join(summarized_text.split())

        # Additional post-processing for spacing issues
        summarized_text = re.sub(r'\s([.,;!?])', r'\1', summarized_text)  # Correct spacing around punctuation marks
        summarized_text = re.sub(r'(\w)\.', r'\1. ', summarized_text)  # Ensure space after a period

        # Append the summary and PDF URL to the list
        summaries.append({'Index': index, 'Summary': summarized_text, 'PDF URL': pdf_url})

        # Print a message after generating the summary
        print(f'Summarized Index {index}')
    except Exception as e:
        # Handle exceptions (e.g., index errors) and skip the problematic file
        print(f"Skipped processing Index {index} due to an error:", str(e))
        continue

# Create a DataFrame from the list of summaries
summary_df = pd.DataFrame(summaries)

# Save the summaries to a CSV file
summary_df.to_csv(output_csv, index=False)

print("Summaries saved to", output_csv)
