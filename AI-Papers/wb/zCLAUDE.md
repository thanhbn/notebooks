# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI research papers repository containing academic papers (PDF/TXT format) on code generation, language models, and software engineering. The primary focus is on collecting and organizing research papers related to code LLMs, benchmarking, and AI-assisted code development.

## Key Tools and Commands

### PDF Processing
```bash
# Convert all PDFs to TXT files
python pdf_to_txt_converter.py --convert

# Rename existing TXT files to include paper titles
python pdf_to_txt_converter.py --rename

# Convert PDFs and rename TXT files
python pdf_to_txt_converter.py --both
```

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv_ai_papers
source venv_ai_papers/bin/activate  # On macOS/Linux
# or
venv_ai_papers\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Research Utilities
```bash
# Search for specific terms across all papers
grep -r "benchmark" *.txt > all_benchmark.md

# List all papers by topic
ls -la *.pdf | grep -E "(LLM|Code|Benchmark)"
```

## Repository Structure

### Core Components
- **pdf_to_txt_converter.py**: Main utility for converting PDFs to searchable text format
- **requirements.txt**: Python dependencies for PDF processing
- **all_benchmark.md**: Comprehensive search results for benchmarking terms across papers

### Research Papers Organization
Papers are organized by research area:
- **1.2.x**: Dataset and evaluation papers (LeetCode, benchmarks)
- **1.3.x**: Code generation models (LLaMA-Reviewer, AgentCoder, StarCoder)
- **1.4.x**: Code understanding and structure (GraphCodeBERT, AST-T5)
- **1.6.x**: Code review and automation
- **2xxx.xxxxx**: ArXiv papers with standard naming convention

### Dependencies
The repository uses lightweight PDF processing libraries:
- **pypdf**, **pymupdf**, **pdfplumber**: PDF text extraction
- **beautifulsoup4**, **requests**: Web scraping and HTML processing
- **pathlib2**: Enhanced path handling

## PDF Processing Architecture

The `pdf_to_txt_converter.py` script implements a robust multi-method PDF processing pipeline:

1. **Primary Method**: PyPDF2 for standard text extraction
2. **Fallback Method**: pdfplumber for complex layouts and tables
3. **File Naming**: Automatic renaming of ArXiv format files to include paper titles
4. **Error Handling**: Graceful handling of corrupted PDFs and encoding issues

### Key Features
- Supports both interactive and command-line modes
- Automatically detects ArXiv paper format (YYMM.NNNNN pattern)
- Extracts paper titles from first page content
- Handles special characters in filenames
- Preserves original PDFs while creating searchable TXT versions

## Research Workflow

### Adding New Papers
1. Place PDF files in the root directory
2. Run `python pdf_to_txt_converter.py --both` to convert and organize
3. Generated TXT files enable full-text search across the research corpus

### Content Analysis
- Use `grep` commands to search across all TXT files
- The `all_benchmark.md` file demonstrates comprehensive search results
- Papers cover benchmarking, code generation, and LLM evaluation

## Working with the Repository

### Text Search and Analysis
- All papers are converted to TXT format for efficient searching
- Use standard Unix tools (grep, find, sort) for content analysis
- The repository supports comparative analysis across different research areas

### Cross-Platform Compatibility
- Script automatically detects Windows vs Unix-like systems
- Uses appropriate path handling for each platform
- Supports various PDF encoding formats and international characters