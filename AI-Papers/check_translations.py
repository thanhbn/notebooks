#!/usr/bin/env python3
import os
import glob

def find_files_needing_translation():
    """Find all .txt files that don't have corresponding _vi.txt translations"""
    
    # Get all .txt files (excluding _vi.txt and requirements.txt)
    all_txt_files = []
    for file in glob.glob('*.txt'):
        if not file.endswith('_vi.txt') and file != 'requirements.txt':
            all_txt_files.append(file)
    
    # Check which ones don't have Vietnamese translations
    missing_translations = []
    for txt_file in all_txt_files:
        # Create the expected Vietnamese filename
        vi_filename = txt_file.replace('.txt', '_vi.txt')
        
        # Check if the Vietnamese version exists
        if not os.path.exists(vi_filename):
            missing_translations.append(txt_file)
    
    # Sort the results
    missing_translations.sort()
    
    return missing_translations

if __name__ == "__main__":
    missing = find_files_needing_translation()
    
    print('Files that need Vietnamese translation:')
    print('=' * 50)
    for i, file in enumerate(missing, 1):
        print(f'{i:2d}. {file}')
    
    print(f'\nTotal files needing translation: {len(missing)}')