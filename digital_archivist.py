import os
import csv
import json
import sys

import ollama
import pymupdf
import docx

MODEL_NAME = "phi4-reasoning:14b-plus-fp16"


def ollama_chat(system_prompt, user_prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ""


def extract_text(path):
    if path.lower().endswith(".pdf"):
        try:
            with pymupdf.open(path) as doc:
                text = "".join(page.get_text("text", sort=True) + "\n" for page in doc)
                return text
        except Exception as e:
            return ""
    elif path.lower().endswith(".docx"):
        try:
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    else:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        return ""


def parse_metadata(text):
    prompt = (
        "Identify author(s), title, publication date, and primary subject matter from the text. "
        "Return JSON with keys 'authors', 'title', 'date', 'subject'."
    )
    result = ollama_chat(prompt, text[:15000])
    try:
        return json.loads(result)
    except Exception:
        return {"authors": "", "title": "", "date": "", "subject": ""}


def summarize(text):
    prompt = "Summarize the document in exactly three concise bullet points."
    response = ollama_chat(prompt, text[:15000])
    if response:
        bullets = [line.strip("- ") for line in response.splitlines() if line.strip()]
        return bullets[:3]
    return []


def generate_tags(summary):
    joined = " ".join(summary)
    prompt = (
        "Based on the summary, generate 3-5 useful topic tags in kebab-case. Return only a comma-separated list."
    )
    result = ollama_chat(prompt, joined)
    return [t.strip() for t in result.split(",") if t.strip()] if result else []


def detect_language(text):
    prompt = "Respond with 'en' if the text is English or 'zh' if it is Mandarin Chinese."
    lang = ollama_chat(prompt, text[:2000]).lower()
    return "zh" if "zh" in lang or "chinese" in lang else "en"


def generate_filename(authors, title, date, language, ext):
    parts = [p for p in [authors, title, date, language] if p]
    name = " - ".join(parts)
    return f"{name}{ext}"


def phase1(directory):
    print("Beginning Phase 1: Analysis & Metadata Generation")
    records = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                path = os.path.join(root, f)
                text = extract_text(path)
                if not text:
                    records.append([path, f, "", "", "", "", "Extraction Failed"])
                    continue
                meta = parse_metadata(text)
                summary = summarize(text)
                tags = generate_tags(summary)
                language = detect_language(text)
                ext = os.path.splitext(f)[1]
                filename = generate_filename(meta.get("authors"), meta.get("title"), meta.get("date"), language, ext)
                records.append([
                    path,
                    f,
                    " | ".join(summary),
                    ",".join(tags),
                    filename,
                    language,
                    "Analyzed",
                ])
    with open("master_index.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "File Path",
            "File Name",
            "Generated Summary",
            "Metadata Tags",
            "New Standardized Name",
            "Language",
            "Processing Status",
        ])
        writer.writerows(records)


def phase2():
    print("Beginning Phase 2: Architecture Synthesis")
    tags = {}
    with open("master_index.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Processing Status"] != "Analyzed":
                continue
            tag_list = [t for t in row["Metadata Tags"].split(",") if t]
            if tag_list:
                tag = tag_list[0]
                tags.setdefault(tag, []).append(row["File Path"])
    with open("folder_structure.json", "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2)


def phase3():
    print("Beginning Phase 3: Final Plan Generation")
    with open("folder_structure.json", encoding="utf-8") as f:
        structure = json.load(f)
    with open("master_index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = []
    for row in rows:
        if row["Processing Status"] != "Analyzed":
            continue
        tag_list = [t for t in row["Metadata Tags"].split(",") if t]
        if not tag_list:
            continue
        folder = tag_list[0]
        new_name = row["New Standardized Name"]
        lines.append(f"mkdir -p '{folder}'")
        lines.append(f"mv '{row['File Path']}' '{os.path.join(folder, new_name)}'")
    with open("execution_plan.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def phase4():
    print("Beginning Phase 4: Confirmation and Execution")
    confirm = input("The final reorganization plan is in execution_plan.md. Shall I proceed with creating directories and moving files? [y/N] ")
    if confirm.lower() != "y":
        print("Execution aborted by user. The 'Digital Archivist' mission is complete.")
        return
    print("Confirmation received. Executing plan.")
    with open("execution_plan.md", encoding="utf-8") as f:
        commands = [line.strip() for line in f if line.strip()]
    for cmd in commands:
        if cmd.startswith("mkdir"):
            path = cmd.split(" ", 1)[1].strip("'")
            os.makedirs(path, exist_ok=True)
        elif cmd.startswith("mv"):
            parts = cmd.split(" ")
            src = parts[1].strip("'")
            dest = parts[2].strip("'")
            os.rename(src, dest)
    print("Reorganization complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python digital_archivist.py <directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    phase1(target_dir)
    phase2()
    phase3()
    phase4()
