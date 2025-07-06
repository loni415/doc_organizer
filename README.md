# Document Organizer

This project provides document processing tools powered by [Ollama](https://ollama.com/) for local large language models. It can analyze documents, generate summaries and topic tags, detect language, and build standardized file names. Results are stored in `master_index.csv`.

## Requirements
- Python 3.9+
- [Ollama](https://ollama.com/) installed with a compatible model
- `pymupdf`, `python-docx`, `ollama`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Process a single document:
```bash
python document_engine.py path/to/document.pdf
```

Run the multi-phase Digital Archivist workflow on a directory:
```bash
python digital_archivist.py path/to/folder
```

Both scripts append results to `master_index.csv`.
