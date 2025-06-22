#!/usr/bin/env bash
## html2md_pandoc.sh  –  chuyển *.html → *.md cho LangChain API
## Path gốc do bạn cung cấp: /mnt/d/llm/langchain/docs/api_reference
#
#set -euo pipefail
#ROOT="/mnt/d/llm/langchain/docs/api_reference"
#
## 1. Cài pandoc (nếu chưa có)
##   sudo apt update && sudo apt install pandoc
#
## 2. Bắt đầu chuyển đổi
#find "$ROOT" -type f -name '*.html' -print0 |
#while IFS= read -r -d '' file; do
#    md="${file%.html}.md"
#        echo "↻ $(realpath --relative-to="$ROOT" "$file") → ${md##*/}"
#            pandoc "$file" -f html -t gfm \
#                       --wrap=none \
#                                  --embed-resources \
#                                             -o "$md"
#                                             done
#
#                                             # 3. Sửa link nội bộ .html → .md (tuỳ chọn)
#                                             python3 - <<'PY'
#                                             import pathlib, re
#                                             root = pathlib.Path("/mnt/d/llm/langchain/docs/api_reference")
#                                             for md in root.rglob("*.md"):
#                                                 txt = md.read_text(encoding="utf-8")
#                                                     txt = re.sub(r'\(([^)]+?)\.html\)', r'(\1.md)', txt)
#                                                         md.write_text(txt, encoding="utf-8")
#                                                         PY
#
