import os
import csv
import json

import ollama
import pymupdf
import docx

MODEL_NAME = "phi4-reasoning:14b-plus-fp16"


def ollama_chat(system_prompt, user_prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""


def extract_text(path):
    if path.lower().endswith(".pdf"):
        try:
            with pymupdf.open(path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text("text", sort=True) + "\n"
                return text
        except Exception as e:
            return f"Error processing PDF: {e}"
    elif path.lower().endswith(".docx"):
        try:
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"Error processing docx file: {e}"
    else:
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return "Error: Could not decode the file."


def parse_metadata(text):
    prompt = (
        "You are a document analysis engine. Identify the author(s), title, publication date, "
        "and primary subject matter from the text. Return a JSON object with keys 'authors', "
        "'title', 'date', and 'subject'. Use YYYY-MM-DD for the date if possible."
    )
    result = ollama_chat(prompt, text[:15000])
    try:
        return json.loads(result)
    except Exception:
        return {"authors": "", "title": "", "date": "", "subject": ""}


def summarize(text):
    prompt = (
        "You are a research analyst. Summarize the document in exactly three concise bullet points."
    )
    response = ollama_chat(prompt, text[:15000])
    if response:
        bullets = [line.strip("- ") for line in response.splitlines() if line.strip()]
        return bullets[:3]
    return []


def generate_tags(summary):
    joined = " ".join(summary)
    prompt = (
        "You are a metadata specialist. Based on the summary, generate 3-5 useful topic tags in kebab-case. "
        "Return only a comma-separated list."
    )
    result = ollama_chat(prompt, joined)
    if result:
        return [t.strip() for t in result.split(",") if t.strip()]
    return []


def detect_language(text):
    prompt = (
        "Identify the primary language of the text. Respond with 'en' for English or 'zh' for Mandarin Chinese."
    )
    lang = ollama_chat(prompt, text[:2000]).lower()
    return "zh" if "zh" in lang or "chinese" in lang else "en"


def generate_filename(authors, title, date, language, ext):
    parts = []
    if authors:
        parts.append(authors)
    if title:
        parts.append(title)
    if date:
        parts.append(date)
    parts.append(language)
    name = " - ".join(parts)
    return f"{name}{ext}"


def append_to_csv(data, csv_path="master_index.csv"):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "File ID",
                "File Path",
                "File Name",
                "Generated Summary",
                "Metadata Tags",
                "New Standardized Name",
                "Language",
                "Processing Status",
            ])
        writer.writerow([
            "",
            "",
            "",
            " | ".join(data["summary"]),
            ",".join(data["tags"]),
            data["new_standardized_name"],
            data["language"],
            "Completed",
        ])


def process_document(path):
    text = extract_text(path)
    meta = parse_metadata(text)
    summary = summarize(text)
    tags = generate_tags(summary)
    language = detect_language(text)
    ext = os.path.splitext(path)[1]
    filename = generate_filename(meta.get("authors", ""), meta.get("title", ""), meta.get("date", ""), language, ext)
    result = {
        "summary": summary,
        "tags": tags,
        "language": language,
        "new_standardized_name": filename,
    }
    append_to_csv(result)
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_engine.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = process_document(file_path)
    print(json.dumps(data, indent=2))
