#!/usr/bin/env python3
"""html2md_pandoc_recursive.py

Recursively convert every .html file inside a ROOT folder to .md using Pandoc.
Adds clear progress output so you can see nested folders being visited.

Usage:
  python html2md_pandoc_recursive.py \
         --root /mnt/d/llm/langchain/docs/api_reference \
         --wrap none --embed

Options:
  --root   Folder that contains the mirrored LangChain docs.
  --wrap   Pandoc wrap mode [none|auto|preserve] (default: none)
  --embed  Embed resources into Markdown.
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path

def run_pandoc(html_path: Path, wrap: str, embed: bool):
    md_path = html_path.with_suffix('.md')
    cmd = [
        'pandoc', str(html_path),
        '-f', 'html', '-t', 'gfm',
        f'--wrap={wrap}',
        '-o', str(md_path)
    ]
    if embed:
        cmd.append('--embed-resources')

    subprocess.run(cmd, check=True)
    return md_path

def convert_all(root: Path, wrap: str, embed: bool):
    html_files = sorted(root.rglob('*.html'))   # rglob is recursive
    if not html_files:
        sys.exit(f'No .html files found under {root}')

    total = len(html_files)
    print(f'Found {total} HTML files. Converting...')
    for idx, html_file in enumerate(html_files, 1):
        rel = html_file.relative_to(root)
        print(f'[{idx}/{total}] {rel}')
        run_pandoc(html_file, wrap, embed)

    print('Fixing internal links (.html → .md)...')
    fix_links(root)
    print('✅ Done.')

def fix_links(root: Path):
    pattern = re.compile(r'\(([^)]+?)\.html\)')
    for md in root.rglob('*.md'):
        content = md.read_text(encoding='utf-8')
        new_content = pattern.sub(r'(\1.md)', content)
        md.write_text(new_content, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description='Convert LangChain HTML docs to Markdown recursively.')
    parser.add_argument('--root', required=True, help='Root directory of the HTML docs')
    parser.add_argument('--wrap', default='none', choices=['none', 'auto', 'preserve'])
    parser.add_argument('--embed', action='store_true')
    args = parser.parse_args()

    root_path = Path(args.root).expanduser()
    if not root_path.exists():
        sys.exit(f'Root path {root_path} does not exist')

    convert_all(root_path, args.wrap, args.embed)

if __name__ == '__main__':
    main()
