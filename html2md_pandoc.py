#!/usr/bin/env python3
"""
html2md_pandoc.py

Convert all .html files inside a ROOT directory (default
/mnt/d/llm/langchain/docs/api_reference) to Markdown (.md) using Pandoc.

Usage:
    python html2md_pandoc.py [-r ROOT] [-w WRAP] [-e]

Options:
    -r --root   Root folder containing .html files (default shown above)
    -w --wrap   Pandoc wrap mode: none, auto, preserve  [default: none]
    -e --embed  Embed images/resources into Markdown via Pandoc

Pandoc must be installed and available in PATH.
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path


def convert_one(html_path: Path, wrap: str, embed: bool):
    """Run Pandoc on a single HTML file -> MD"""
    md_path = html_path.with_suffix('.md')
    cmd = [
        'pandoc',
        str(html_path),
        '-f', 'html',
        '-t', 'gfm',
        f'--wrap={wrap}',
    ]
    if embed:
        cmd.append('--embed-resources')
    cmd.extend(['-o', str(md_path)])

    print(f'↻ {html_path} → {md_path}')
    subprocess.run(cmd, check=True)


def fix_links(root: Path):
    """Rewrite internal .html links to .md"""
    pattern = re.compile(r'\(([^)]+?)\.html\)')
    for md in root.rglob('*.md'):
        text = md.read_text(encoding='utf-8')
        new_text = pattern.sub(r'(\1.md)', text)
        md.write_text(new_text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Convert LangChain API HTML to Markdown')
    parser.add_argument('-r', '--root',
                        default='/mnt/d/llm/langchain/docs/api_reference',
                        help='Root directory containing HTML files')
    parser.add_argument('-w', '--wrap', default='none',
                        choices=['none', 'auto', 'preserve'],
                        help='Pandoc wrap mode')
    parser.add_argument('-e', '--embed', action='store_true',
                        help='Embed resources (images) into Markdown')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f'Error: root path {root} does not exist')

    html_files = list(root.rglob('*.html'))
    if not html_files:
        sys.exit('Error: no .html files found under the specified root')

    print(f'Found {len(html_files)} HTML files under {root}')
    for html in html_files:
        convert_one(html, args.wrap, args.embed)

    fix_links(root)
    print('✅ Conversion complete.')


if __name__ == '__main__':
    main()
