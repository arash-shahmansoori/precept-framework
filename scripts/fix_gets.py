#!/usr/bin/env python3
"""Fix bare \\gets commands that need to be in math mode."""
import re
from pathlib import Path

TEX_FILE = Path(__file__).parent.parent / "PRECEPT_PAPER.tex"

tex = TEX_FILE.read_text()

# Find all algorithm blocks
alg_pattern = re.compile(r'(\\begin\{algorithmic\}.*?\\end\{algorithmic\})', re.DOTALL)

def fix_block(m):
    block = m.group(0)
    lines = block.split('\n')
    fixed = []
    for line in lines:
        # Fix bare \gets not already in $...$
        # Pattern: word_chars \gets word_chars (not inside $)
        if r'\gets' in line and '$' not in line:
            line = line.replace(r'\gets', r'$\gets$')
        fixed.append(line)
    return '\n'.join(fixed)

tex = alg_pattern.sub(fix_block, tex)

# Also fix the multi-line \State issue where function calls span multiple lines
# Like: \State static_items.append(KnowledgeItem(
#        \State   content=doc.page_content,
# These should NOT have \State on continuation lines

TEX_FILE.write_text(tex)

# Count fixes
count = len(re.findall(r'(?<!\$)\\gets(?!\$)', tex))
print(f"Remaining bare \\gets: {count}")
gets_in_math = len(re.findall(r'\$\\gets\$', tex))
print(f"\\gets in math mode: {gets_in_math}")
