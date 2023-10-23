import os
import pdfplumber
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
from pdfminer.pdfdocument import PDFSyntaxError
import re

# Path to add PDF files for summary
pdf_directory = 'data/First 300'

# Output CSV file of summaries
output_csv = 'summaries_300.csv'

# Initializing BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

summaries = []

# Regular expression pattern to remove special characters
special_chars_pattern = r'[^a-zA-Z\s]+'
square_brackets_pattern = r'\[[^\]]+\]'

for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)

        try:
            # Extract text from the entire PDF
            with pdfplumber.open(pdf_path) as pdf:
                extracted_text = ''
                for page in pdf.pages:
                    extracted_text += page.extract_text()

            # Tokenize the extracted text
            inputs = tokenizer(extracted_text, return_tensors='pt', max_length=1024, truncation=True)

            # Summary generation
            summary_ids = model.generate(inputs['input_ids'], num_beams=5, min_length=150, max_length=350, early_stopping=True)
            summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Remove special characters, numbers, and text enclosed in square brackets
            summarized_text = re.sub(special_chars_pattern, ' ', summarized_text)
            summarized_text = re.sub(square_brackets_pattern, '', summarized_text)
            summarized_text = ' '.join(word for word in summarized_text.split() if not word.isdigit())  # Remove numbers

            # Normalize text by removing extra spaces
            summarized_text = ' '.join(summarized_text.split())

            # Additional post-processing for spacing issues
            # Replace problematic spacing patterns
            summarized_text = re.sub(r'\s([.,;!?])', r'\1', summarized_text)  # Correct spacing around punctuation marks
            summarized_text = re.sub(r'(\w)\.', r'\1. ', summarized_text)  # Ensure space after period

            # Append the summary to the list
            summaries.append({'Filename': filename, 'Summary': summarized_text})

            # Print a message after summarizing the PDF
            print(f'Summarized {filename}')

        except PDFSyntaxError:
            print(f"Skipped processing {filename} due to PDFSyntaxError.")
            continue

# Create a DataFrame from the list of summaries
df = pd.DataFrame(summaries)

# Save the summaries to a CSV file
df.to_csv(output_csv, index=False)

print("Summaries saved to", output_csv)
