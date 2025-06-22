#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF to TXT Converter
Convert all PDF files in directory to TXT format
"""

import os
import glob
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

def main():
    """
    Main function to convert all PDFs in directory
    """
    # Working directory
    working_dir = r"D:\llm\notebooks\AI-Papers"
    
    print("=" * 60)
    print("PDF TO TXT CONVERTER")
    print("=" * 60)
    print(f"Working directory: {working_dir}")
    
    # Find all PDF files
    pdf_pattern = os.path.join(working_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print("No PDF files found in directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
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

if __name__ == "__main__":
    main()
