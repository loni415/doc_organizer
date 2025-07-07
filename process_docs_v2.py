#!/usr/bin/env python3
"""
Document Organizer V2 - Combined Analysis and Organization Tool

Combines the analysis capabilities of process_documents2.py with the 
reorganization features of digital_archivist.py into a unified tool.

Features:
- Comprehensive document analysis (summary, tags, language, metadata)
- Flexible output options (analysis-only or full reorganization)
- Robust error handling and progress tracking
- Multiple workflow modes (analyze, reorganize, or both)
"""

import os
import sys
import json
import csv
import argparse
import shutil  # Add this import
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymupdf  # PyMuPDF
import docx
import ollama

# --- CONFIGURATION ---
MODEL_NAME = "phi4-reasoning:14b-plus-fp16"

# --- CORE UTILITY FUNCTIONS ---

def ollama_chat_request(system_prompt, user_prompt):
    """Generic function to handle requests to Ollama and get responses."""
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

# --- TEXT EXTRACTION FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a local PDF file."""
    try:
        with pymupdf.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
            return text
    except Exception as e:
        return f"Error processing PDF: {e}"

def extract_text_from_docx(file_path):
    """Extracts text from a local .docx file."""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error processing docx file: {e}"

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

def extract_text(file_path):
    """Universal text extraction function that routes to appropriate handler."""
    file_name = os.path.basename(file_path).lower()
    
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_name.endswith(('.txt', '.md')):
        return extract_text_from_txt(file_path)
    else:
        return f"Error: Unsupported file type"

# --- ANALYSIS FUNCTIONS ---

def get_summary(text_content):
    """Generate a concise summary of the document."""
    system_prompt = "You are a research analyst. Summarize the following text in 3 concise bullet points. Be neutral and factual."
    return ollama_chat_request(system_prompt, text_content[:15000])

def get_tags(summary):
    """Extract topic tags from the summary."""
    system_prompt = "You are a metadata specialist. Based on the following summary, generate a list of 3-5 relevant topic tags. The tags should be specific and useful for categorization. Return ONLY a comma-separated list of tags, with no other text."
    user_prompt = f"Summary:\n{summary}"
    tags_string = ollama_chat_request(system_prompt, user_prompt)
    if tags_string:
        return [tag.strip() for tag in tags_string.split(',')]
    return []

def get_language(text_content):
    """Detect the primary language of the document."""
    system_prompt = "Identify the primary language of the following text. Respond with ONLY the name of the language (e.g., 'English', 'Mandarin Chinese')."
    return ollama_chat_request(system_prompt, text_content[:2000])

def parse_metadata(text_content):
    """Extract structured metadata from document text."""
    system_prompt = (
        "Identify author(s), title, publication date, and primary subject matter from the text. "
        "Return JSON with keys 'authors', 'title', 'date', 'subject'."
    )
    result = ollama_chat_request(system_prompt, text_content[:15000])
    try:
        return json.loads(result) if result else {}
    except json.JSONDecodeError:
        return {"authors": "", "title": "", "date": "", "subject": ""}

def generate_filename(tags, metadata=None, original_filename="", use_metadata=False):
    """Generate a standardized filename based on tags and optionally metadata."""
    if not tags: 
        return ""
    
    # Preserve original file extension
    original_ext = os.path.splitext(original_filename)[1] if original_filename else ".ext"
    
    if use_metadata and metadata:
        # Use metadata-based naming (digital_archivist style)
        parts = []
        if metadata.get("authors"):
            parts.append(metadata["authors"])
        if metadata.get("title"):
            parts.append(metadata["title"])
        if metadata.get("date"):
            parts.append(metadata["date"])
        
        if parts:
            name = " - ".join(parts)
            return f"{name}{original_ext}"
    
    # Use tag-based naming (process_documents2 style)
    system_prompt = f"You are a file naming expert. Based on the following tags, create a standardized filename. The format is YYYY-MM-DD_[Primary-Tag]_[Keywords]{original_ext}. Use the most important tag first. Respond with ONLY the filename."
    user_prompt = f"Tags: {', '.join(tags)}"
    filename = ollama_chat_request(system_prompt, user_prompt)
    return filename or f"{datetime.now().strftime('%Y-%m-%d')}_{tags[0]}{original_ext}"

# --- WORKFLOW FUNCTIONS ---

def analyze_documents(folder_path, output_csv="analysis_results.csv", include_metadata=False):
    """
    Analyze documents in a folder and save results to CSV.
    
    Args:
        folder_path: Path to folder containing documents
        output_csv: Output CSV filename
        include_metadata: Whether to extract detailed metadata
    """
    print(f"Starting document analysis for: {folder_path}")
    
    # Collect all files
    file_data = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
                file_data.append({
                    'File Path': os.path.join(root, f), 
                    'File Name': f
                })
    
    if not file_data:
        print("No supported files found.")
        return None
        
    df = pd.DataFrame(file_data)
    print(f"[INFO] Found {len(df)} total files to analyze.")
    
    # Initialize columns
    columns = ['Generated Summary', 'Metadata Tags', 'New Standardized Name', 'Language', 'Processing Status']
    if include_metadata:
        columns.extend(['Authors', 'Title', 'Date', 'Subject'])
    
    for col in columns:
        if col not in df.columns:
            df[col] = ''

    # Process each file
    for index, row in df.iterrows():
        file_path, file_name = row['File Path'], row['File Name']
        print(f"\nProcessing ({index + 1}/{len(df)}): {file_name}")

        # Extract text
        text_content = extract_text(file_path)
        
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
            
            # Extract metadata if requested
            metadata = {}
            if include_metadata:
                print("  [3.5/4] Extracting metadata...")
                metadata = parse_metadata(text_content)
                for key in ['Authors', 'Title', 'Date', 'Subject']:
                    df.at[index, key] = metadata.get(key.lower(), '')
            
            print("  [4/4] Generating standardized filename...")
            new_filename = generate_filename(tags, metadata, str(file_name), include_metadata)
            if new_filename:
                df.at[index, 'New Standardized Name'] = new_filename
            
            df.at[index, 'Processing Status'] = 'Analyzed'
            print(f"  [SUCCESS] Analysis complete for {file_name}")
            
        except Exception as e:
            print(f"  [ERROR] Analysis failed for {file_name}: {e}")
            df.at[index, 'Processing Status'] = f'Error: {e}'

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Analysis complete. Results saved to {output_csv}")
    return df

def create_folder_structure(csv_file="analysis_results.csv", structure_file="folder_structure.json"):
    """Create folder structure based on document tags."""
    print("Creating folder structure based on tags...")
    
    tags_to_files = {}
    
    try:
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            if row['Processing Status'] != 'Analyzed':
                continue
                
            tags = [t.strip() for t in str(row['Metadata Tags']).split(',') if t.strip() and str(row['Metadata Tags']) != 'nan']
            if tags:
                primary_tag = tags[0]  # Use first tag as primary folder
                if primary_tag not in tags_to_files:
                    tags_to_files[primary_tag] = []
                tags_to_files[primary_tag].append(str(row['File Path']))  # Convert to string
        
        # Save structure
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(tags_to_files, f, indent=2, ensure_ascii=False)
        
        print(f"Folder structure saved to {structure_file}")
        print(f"Found {len(tags_to_files)} categories for organization")
        
        return tags_to_files
        
    except Exception as e:
        print(f"Error creating folder structure: {e}")
        return {}

def generate_execution_plan(csv_file="analysis_results.csv", plan_file="execution_plan.md"):
    """Generate shell commands for file reorganization."""
    print("Generating execution plan...")
    
    commands = []
    
    try:
        df = pd.read_csv(csv_file)
        
        # Group by primary tag
        folder_commands = set()
        move_commands = []
        
        for _, row in df.iterrows():
            if row['Processing Status'] != 'Analyzed':
                continue
                
            tags = [t.strip() for t in str(row['Metadata Tags']).split(',') if t.strip() and str(row['Metadata Tags']) != 'nan']
            if not tags:
                continue
            
            folder = tags[0]  # Primary tag becomes folder name
            new_name = str(row['New Standardized Name'])  # Convert to string
            old_path = str(row['File Path'])  # Convert to string
            
            # Sanitize folder name for filesystem
            folder = "".join(c for c in folder if c.isalnum() or c in (' ', '-', '_')).strip()
            folder = folder.replace(' ', '_')
            
            folder_commands.add(f"mkdir -p '{folder}'")
            new_path = os.path.join(folder, new_name)
            move_commands.append(f"mv '{old_path}' '{new_path}'")
        
        # Combine commands
        commands = list(folder_commands) + move_commands
        
        # Save plan
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write("# Document Reorganization Plan\n\n")
            f.write("## Commands to execute:\n\n")
            f.write("```bash\n")
            f.write('\n'.join(commands))
            f.write("\n```\n")
        
        print(f"Execution plan saved to {plan_file}")
        print(f"Generated {len(commands)} commands")
        
        return commands
        
    except Exception as e:
        print(f"Error generating execution plan: {e}")
        return []

def execute_reorganization(plan_file="execution_plan.md", dry_run=False):
    """Execute the reorganization plan."""
    if not os.path.exists(plan_file):
        print(f"Error: Plan file {plan_file} not found")
        return False
    
    print(f"Reading execution plan from {plan_file}")
    
    # Read commands from plan file
    commands = []
    with open(plan_file, 'r', encoding='utf-8') as f:
        in_code_block = False
        for line in f:
            line = line.strip()
            if line == "```bash":
                in_code_block = True
                continue
            elif line == "```":
                in_code_block = False
                continue
            elif in_code_block and line:
                commands.append(line)
    
    if not commands:
        print("No commands found in plan file")
        return False
    
    print(f"Found {len(commands)} commands to execute")
    
    if dry_run:
        print("\n--- DRY RUN MODE ---")
        for cmd in commands:
            print(f"Would execute: {cmd}")
        return True
    
    # Ask for confirmation
    print("\n--- EXECUTION PLAN PREVIEW ---")
    for cmd in commands[:5]:  # Show first 5 commands
        print(f"  {cmd}")
    if len(commands) > 5:
        print(f"  ... and {len(commands) - 5} more commands")
    
    confirm = input(f"\nExecute {len(commands)} commands? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Execution cancelled by user")
        return False
    
    # Execute commands
    print("Executing reorganization plan...")
    
    try:
        for i, cmd in enumerate(commands, 1):
            print(f"  ({i}/{len(commands)}) {cmd}")
            
            if cmd.startswith("mkdir"):
                # Extract directory path
                path = cmd.split("mkdir -p ", 1)[1].strip("'\"")
                os.makedirs(path, exist_ok=True)
                
            elif cmd.startswith("mv"):
                # Extract source and destination
                parts = cmd.split("'")
                if len(parts) >= 4:
                    src = parts[1]
                    dest = parts[3]
                    
                    # Validate source file exists
                    if not os.path.exists(src):
                        print(f"Warning: Source file not found: {src}")
                        continue
                    
                    # Ensure destination directory exists
                    dest_dir = os.path.dirname(dest)
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    
                    # Move file
                    shutil.move(src, dest)
                else:
                    print(f"Warning: Could not parse mv command: {cmd}")
        
        print("Reorganization completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return False

# --- MAIN WORKFLOW MODES ---

def mode_analyze_only(args):
    """Analysis-only mode: analyze documents and save results."""
    print("=== ANALYSIS MODE ===")
    
    df = analyze_documents(
        folder_path=args.folder,
        output_csv=args.output,
        include_metadata=args.metadata
    )
    
    if df is not None:
        # Print summary
        total = len(df)
        analyzed = len(df[df['Processing Status'] == 'Analyzed'])
        print(f"\n=== SUMMARY ===")
        print(f"Total files: {total}")
        print(f"Successfully analyzed: {analyzed}")
        print(f"Results saved to: {args.output}")

def mode_reorganize(args):
    """Full reorganization mode: analyze, plan, and execute."""
    print("=== REORGANIZATION MODE ===")
    
    # Step 1: Analysis
    print("\n--- Step 1: Document Analysis ---")
    df = analyze_documents(
        folder_path=args.folder,
        output_csv="master_index.csv",
        include_metadata=True
    )
    
    if df is None:
        return
    
    # Step 2: Create folder structure
    print("\n--- Step 2: Architecture Synthesis ---")
    structure = create_folder_structure("master_index.csv", "folder_structure.json")
    
    if not structure:
        print("No valid structure created. Aborting.")
        return
    
    # Step 3: Generate execution plan
    print("\n--- Step 3: Plan Generation ---")
    commands = generate_execution_plan("master_index.csv", "execution_plan.md")
    
    if not commands:
        print("No execution plan generated. Aborting.")
        return
    
    # Step 4: Execute (with confirmation)
    print("\n--- Step 4: Execution ---")
    success = execute_reorganization("execution_plan.md", dry_run=args.dry_run)
    
    if success:
        print("\n=== REORGANIZATION COMPLETE ===")
    else:
        print("\n=== REORGANIZATION FAILED OR CANCELLED ===")

# --- COMMAND LINE INTERFACE ---

def main():
    parser = argparse.ArgumentParser(
        description="Document Organizer V2 - Analyze and organize documents using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze documents only
  python process_docs_v2.py analyze /path/to/documents --output results.csv
  
  # Full reorganization with confirmation
  python process_docs_v2.py reorganize /path/to/documents
  
  # Dry run (show what would happen)
  python process_docs_v2.py reorganize /path/to/documents --dry-run
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Analyze mode
    analyze_parser = subparsers.add_parser('analyze', help='Analyze documents only')
    analyze_parser.add_argument('folder', help='Folder path containing documents')
    analyze_parser.add_argument('--output', default='analysis_results.csv', 
                               help='Output CSV file (default: analysis_results.csv)')
    analyze_parser.add_argument('--metadata', action='store_true',
                               help='Include detailed metadata extraction')
    
    # Reorganize mode
    reorg_parser = subparsers.add_parser('reorganize', help='Analyze and reorganize documents')
    reorg_parser.add_argument('folder', help='Folder path containing documents')
    reorg_parser.add_argument('--dry-run', action='store_true',
                             help='Show what would happen without executing')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Validate folder
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        sys.exit(1)
    
    # Execute mode
    if args.mode == 'analyze':
        mode_analyze_only(args)
    elif args.mode == 'reorganize':
        mode_reorganize(args)

if __name__ == "__main__":
    main()
