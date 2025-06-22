#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF to TXT Converter
Convert all PDF files in directory to TXT format
"""

import os
import glob
import re
import subprocess
import sys
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file using multiple methods
    """
    text = ""
    
    # Method 1: Try PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print(f"  - Processing {total_pages} pages with PyPDF2...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- PAGE {page_num} ---\n"
                        text += page_text + "\n"
                    else:
                        text += f"\n--- PAGE {page_num} ---\n"
                        text += "[IMAGE: Unable to extract text from this page]\n"
                        
                except Exception as e:
                    text += f"\n--- PAGE {page_num} ---\n"
                    text += f"[ERROR: Cannot process this page - {str(e)}]\n"
                    
        return text
                    
    except ImportError:
        pass
    except Exception as e:
        text = f"[ERROR: Cannot read PDF file - {str(e)}]"
        
    # Method 2: Try pdfplumber as fallback
    try:
        import pdfplumber
        print(f"  - Trying pdfplumber as fallback...")
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n--- PAGE {page_num} ---\n"
                        text += page_text + "\n"
                    else:
                        text += f"\n--- PAGE {page_num} ---\n"
                        text += "[IMAGE: Unable to extract text from this page]\n"
                        
                        # Try to extract tables if text extraction fails
                        tables = page.extract_tables()
                        if tables:
                            text += "[TABLES FOUND - Attempting to extract data]\n"
                            for table_num, table in enumerate(tables, 1):
                                text += f"Table {table_num}:\n"
                                for row in table:
                                    if row:
                                        text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                                text += "\n"
                        
                except Exception as e:
                    text += f"\n--- PAGE {page_num} ---\n"
                    text += f"[ERROR: Cannot process this page - {str(e)}]\n"
                    
        return text
        
    except ImportError:
        text = "[ERROR: No PDF processing library available. Please install PyPDF2 or pdfplumber]"
    except Exception as e:
        text = f"[ERROR: Cannot read PDF file with pdfplumber - {str(e)}]"
        
    return text

def convert_pdf_to_txt(pdf_path, txt_path):
    """
    Convert a PDF file to TXT
    """
    print(f"Converting: {os.path.basename(pdf_path)}")
    
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Create header for TXT file
    header = f"""# {os.path.basename(pdf_path)}
# Converted from PDF to TXT
# Source path: {pdf_path}
# File size: {os.path.getsize(pdf_path)} bytes

===============================================
PDF FILE CONTENT
===============================================

"""
    
    # Write to TXT file
    try:
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(header + extracted_text)
        print(f"  [OK] Created: {os.path.basename(txt_path)}")
        return True
    except Exception as e:
        print(f"  [ERROR] Write error: {str(e)}")
        return False

def is_arxiv_format(filename):
    """
    Check if filename matches arxiv ID format
    Examples: 2105.12655v2.txt, 2402.01035v2.txt, 2004.13820v2.txt
    """
    # Pattern for arxiv IDs: YYMM.NNNNN or YYMM.NNNNNvN
    # Note: Some papers have 4 or 5 digits after the dot
    pattern = r'^\d{4}\.\d{4,5}(v\d+)?\.txt$'
    return bool(re.match(pattern, filename))

def normalize_paper_name(paper_name):
    """
    Normalize paper name by replacing special characters
    - Replace ':' with '-'
    - Replace spaces with '_'
    - Remove other special characters
    """
    # Remove leading/trailing whitespace
    paper_name = paper_name.strip()
    
    # Replace colons with hyphens
    paper_name = paper_name.replace(':', '-')
    
    # Replace spaces with underscores
    paper_name = paper_name.replace(' ', '_')
    
    # Remove or replace other problematic characters
    paper_name = paper_name.replace('/', '-')
    paper_name = paper_name.replace('\\', '-')
    paper_name = paper_name.replace('?', '')
    paper_name = paper_name.replace('*', '')
    paper_name = paper_name.replace('"', '')
    paper_name = paper_name.replace('<', '')
    paper_name = paper_name.replace('>', '')
    paper_name = paper_name.replace('|', '-')
    paper_name = paper_name.replace('\n', '_')
    paper_name = paper_name.replace('\r', '')
    
    # Replace multiple underscores with single underscore
    paper_name = re.sub(r'_+', '_', paper_name)
    
    # Replace multiple hyphens with single hyphen
    paper_name = re.sub(r'-+', '-', paper_name)
    
    # Remove trailing underscores or hyphens
    paper_name = paper_name.rstrip('_-')
    
    # Limit length to avoid filesystem issues
    if len(paper_name) > 100:
        paper_name = paper_name[:100]
    
    return paper_name

def extract_paper_name_from_txt(txt_file):
    """
    Extract paper name from the first page using grep
    """
    try:
        # Read the file and look for PAGE 1 marker
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Find the PAGE 1 marker and get lines after it
        page1_marker = "--- PAGE 1 ---"
        page1_index = content.find(page1_marker)
        
        if page1_index != -1:
            # Get text after PAGE 1 marker
            after_page1 = content[page1_index + len(page1_marker):].strip()
            lines = after_page1.split('\n')
            
            # Look for the paper title in the next few lines
            for i in range(min(10, len(lines))):  # Check up to 10 lines
                line = lines[i].strip()
                # Skip empty lines and obvious non-title lines
                if line and not line.startswith('[') and not line.startswith('#'):
                    # This should be the paper title
                    return line
                    
    except Exception as e:
        print(f"    [ERROR] Failed to extract paper name: {str(e)}")
    
    return None

def rename_arxiv_txt_files(working_dir):
    """
    Rename TXT files that match arxiv ID format to include paper name
    """
    print("\n" + "=" * 60)
    print("RENAMING ARXIV TXT FILES")
    print("=" * 60)
    
    # Find all TXT files
    txt_pattern = os.path.join(working_dir, "*.txt")
    txt_files = glob.glob(txt_pattern)
    
    print(f"\nFound {len(txt_files)} TXT files in total")
    
    renamed_count = 0
    skipped_count = 0
    
    # First, show arxiv format files that will be processed
    arxiv_files = [f for f in txt_files if is_arxiv_format(os.path.basename(f))]
    print(f"Found {len(arxiv_files)} files matching arxiv format")
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        
        # Check if filename matches arxiv format
        if not is_arxiv_format(filename):
            # Only show first few non-arxiv files to avoid clutter
            if skipped_count < 5:
                print(f"Skipping (not arxiv format): {filename}")
            skipped_count += 1
            continue
        
        # Extract arxiv ID (without .txt extension)
        arxiv_id = filename[:-4]  # Remove .txt
        
        # Extract paper name from file content
        paper_name = extract_paper_name_from_txt(txt_path)
        
        if not paper_name:
            print(f"Skipping (no paper name found): {filename}")
            skipped_count += 1
            continue
        
        # Normalize the paper name
        normalized_name = normalize_paper_name(paper_name)
        
        # Create new filename
        new_filename = f"{arxiv_id}-{normalized_name}.txt"
        new_path = os.path.join(working_dir, new_filename)
        
        # Check if new filename already exists
        if os.path.exists(new_path):
            print(f"Skipping (target exists): {filename} -> {new_filename}")
            skipped_count += 1
            continue
        
        # Rename the file
        try:
            os.rename(txt_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
        except Exception as e:
            print(f"Failed to rename {filename}: {str(e)}")
            skipped_count += 1
    
    print(f"\nRenaming complete:")
    print(f"  - Renamed: {renamed_count} files")
    print(f"  - Skipped: {skipped_count} files")

def main():
    """
    Main function to convert all PDFs in directory and rename TXT files
    """
    # Working directory - use current directory or detect from script location
    if os.name == 'nt':  # Windows
        working_dir = r"D:\llm\notebooks\AI-Papers"
    else:  # Linux/WSL
        # Use current directory
        working_dir = os.getcwd()
    
    print("=" * 60)
    print("PDF TO TXT CONVERTER AND RENAMER")
    print("=" * 60)
    print(f"Working directory: {working_dir}")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--convert", "-c"]:
            choice = "1"
            print("\nMode: Convert PDFs only")
        elif arg in ["--rename", "-r"]:
            choice = "2"
            print("\nMode: Rename TXT files only")
        elif arg in ["--both", "-b"]:
            choice = "3"
            print("\nMode: Convert PDFs and rename TXT files")
        else:
            print("\nUsage:")
            print("  python pdf_to_txt_converter.py [option]")
            print("\nOptions:")
            print("  --convert, -c   Convert PDFs to TXT only")
            print("  --rename, -r    Rename existing TXT files only")
            print("  --both, -b      Convert PDFs and then rename all TXT files (default)")
            return
    else:
        # Interactive mode
        print("\nOptions:")
        print("1. Convert PDFs to TXT only")
        print("2. Rename existing TXT files only")
        print("3. Convert PDFs and then rename all TXT files")
        
        choice = input("\nEnter your choice (1/2/3) [default: 3]: ").strip()
        if not choice:
            choice = "3"
    
    if choice in ["1", "3"]:
        # Find all PDF files
        pdf_pattern = os.path.join(working_dir, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            print("\nNo PDF files found in directory!")
        else:
            print(f"\nFound {len(pdf_files)} PDF files:")
            for pdf_file in pdf_files:
                print(f"  - {os.path.basename(pdf_file)}")
            
            print("\n" + "=" * 60)
            print("STARTING CONVERSION")
            print("=" * 60)
            
            successful = 0
            failed = 0
            
            # Process each PDF file
            for pdf_path in pdf_files:
                # Create corresponding TXT file path
                pdf_name = Path(pdf_path).stem  # Get filename without extension
                txt_path = os.path.join(working_dir, f"{pdf_name}.txt")
                
                # Check if TXT file already exists
                if os.path.exists(txt_path):
                    print(f"File {os.path.basename(txt_path)} already exists. Skipping...")
                    successful += 1
                    continue
                
                # Convert PDF to TXT
                if convert_pdf_to_txt(pdf_path, txt_path):
                    successful += 1
                else:
                    failed += 1
                
                print()  # Empty line for readability
            
            # Report results
            print("=" * 60)
            print("CONVERSION RESULTS")
            print("=" * 60)
            print(f"Successful: {successful} files")
            print(f"Failed: {failed} files")
            print(f"Total: {len(pdf_files)} files")
            
            if successful > 0:
                print(f"\nTXT files created in directory: {working_dir}")
    
    if choice in ["2", "3"]:
        # Rename arxiv format files to include paper names
        rename_arxiv_txt_files(working_dir)

if __name__ == "__main__":
    main()
