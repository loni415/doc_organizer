import os
import pandas as pd
import pymupdf  # PyMuPDF
import json
import requests
import ollama
import docx
from datetime import datetime

# --- CONFIGURATION ---
GDRIVE_INPUT_CSV = 'master_index.csv'
GDRIVE_OUTPUT_CSV = 'analyzed_index.csv'
APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbwY3qtUcRp0YZIq6AABteWVMCOcfsBBet93p1Wvy81WFhK2RitvFdpxwXPmRHCsfYuQ/exec'
MODEL_NAME = "phi4-reasoning:14b-plus-fp16" # Recommend starting with a smaller, faster model for this multi-step process

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """IMPROVED: Extracts text from a local PDF, using flags to preserve layout."""
    try:
        with pymupdf.open(pdf_path) as doc:
            # Using 'blocks' gives better structure than raw text
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
            return text
    except Exception as e:
        return f"Error processing PDF: {e}"

def extract_text_from_txt(file_path):
    """Extracts text from a local .txt or .md file, trying multiple encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return f"Error: Could not decode the file."

def extract_text_from_docx(file_path):
    """Extracts text from a local .docx file."""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error processing docx file: {e}"

# --- NEW: MODULAR OLLAMA ANALYSIS FUNCTIONS ---

def ollama_chat_request(system_prompt, user_prompt):
    """Generic function to handle a request to Ollama and get the response."""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"  [!] Ollama request failed: {e}")
        return None

def get_summary(text_content):
    system_prompt = "You are a research analyst. Summarize the following text in 3 concise bullet points. Be neutral and factual."
    return ollama_chat_request(system_prompt, text_content[:15000])

def get_tags(summary):
    system_prompt = "You are a metadata specialist. Based on the following summary, generate a list of 3-5 relevant topic tags. The tags should be specific and useful for categorization. Return ONLY a comma-separated list of tags, with no other text."
    user_prompt = f"Summary:\n{summary}"
    tags_string = ollama_chat_request(system_prompt, user_prompt)
    if tags_string:
        return [tag.strip() for tag in tags_string.split(',')]
    return []

def get_language(text_content):
    system_prompt = "Identify the primary language of the following text. Respond with ONLY the name of the language (e.g., 'English', 'Mandarin Chinese')."
    return ollama_chat_request(system_prompt, text_content[:2000])

def get_filename(tags):
    if not tags: return ""
    system_prompt = "You are a file naming expert. Based on the following tags, create a standardized filename. The format is YYYY-MM-DD_[Primary-Tag]_[Keywords].ext. Use the most important tag first. Respond with ONLY the filename."
    user_prompt = f"Tags: {', '.join(tags)}"
    filename = ollama_chat_request(system_prompt, user_prompt)
    return filename or f"{datetime.now().strftime('%Y-%m-%d')}_{tags[0]}.ext"


# --- WORKFLOW 2: LOCAL FOLDER (REFACTORED) ---

def process_local_folder(folder_path, output_csv):
    """Scans a local folder, analyzes files with a multi-step process, and saves results."""
    print(f"Starting local folder analysis for: {folder_path}")
    
    file_data = [{'File Path': os.path.join(r, f), 'File Name': f} for r, _, files in os.walk(folder_path) for f in files]
    if not file_data: print("No files found."); return
        
    df = pd.DataFrame(file_data)
    print(f"[INFO] Found {len(df)} total files to inspect.")
    for col in ['Generated Summary', 'Metadata Tags', 'New Standardized Name', 'Language', 'Processing Status']:
        if col not in df.columns: df[col] = ''

    for index, row in df.iterrows():
        file_path, file_name = row['File Path'], row['File Name']
        print(f"\nProcessing ({index + 1}/{len(df)}): {file_name}")

        text_content = ""
        # Text extraction block
        if file_name.lower().endswith('.pdf'):
            text_content = extract_text_from_pdf(file_path)
        elif file_name.lower().endswith('.docx'):
            text_content = extract_text_from_docx(file_path)
        elif file_name.lower().endswith(('.txt', '.md')):
            text_content = extract_text_from_txt(file_path)
        else:
            print(f"  [SKIP] Unsupported file type: {file_name}")
            df.at[index, 'Processing Status'] = 'Skipped - Unsupported file type'
            continue

        # Check if text extraction was successful
        if text_content.startswith("Error"):
            print(f"  [ERROR] {text_content}")
            df.at[index, 'Processing Status'] = f'Error: {text_content}'
            continue

        if not text_content.strip():
            print(f"  [SKIP] No text content found in {file_name}")
            df.at[index, 'Processing Status'] = 'Skipped - No text content'
            continue

        # Multi-step analysis
        try:
            print("  [1/4] Generating summary...")
            summary = get_summary(text_content)
            if summary:
                df.at[index, 'Generated Summary'] = summary
            
            print("  [2/4] Extracting tags...")
            tags = get_tags(summary) if summary else []
            if tags:
                df.at[index, 'Metadata Tags'] = ', '.join(tags)
            
            print("  [3/4] Detecting language...")
            language = get_language(text_content)
            if language:
                df.at[index, 'Language'] = language
            
            print("  [4/4] Generating standardized filename...")
            new_filename = get_filename(tags) if tags else ""
            if new_filename:
                df.at[index, 'New Standardized Name'] = new_filename
            
            df.at[index, 'Processing Status'] = 'Completed'
            print(f"  [SUCCESS] Analysis complete for {file_name}")
            
        except Exception as e:
            print(f"  [ERROR] Analysis failed for {file_name}: {e}")
            df.at[index, 'Processing Status'] = f'Error: {e}'

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Analysis complete. Results saved to {output_csv}")