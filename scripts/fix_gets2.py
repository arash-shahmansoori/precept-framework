#!/usr/bin/env python3
"""More precise fix for \\gets - wrap each bare \\gets individually."""
import re
from pathlib import Path

TEX_FILE = Path(__file__).parent.parent / "PRECEPT_PAPER.tex"
tex = TEX_FILE.read_text()

# Replace ALL bare \gets that aren't already wrapped in $...$
# Strategy: Find \gets not immediately preceded/followed by $
# Use negative lookbehind/lookahead

# First pass: find \gets NOT preceded by $ and NOT followed by $
# But we need to be careful - $\gets$ has $ right before \gets
count = 0
result = []
i = 0
while i < len(tex):
    if tex[i:i+5] == r'\gets':
        # Check if preceded by $
        preceded_by_dollar = (i > 0 and tex[i-1] == '$')
        # Check if followed by $
        followed_by_dollar = (i + 5 < len(tex) and tex[i+5] == '$')
        
        if preceded_by_dollar and followed_by_dollar:
            # Already wrapped: $\gets$
            result.append(r'\gets')
            i += 5
        elif not preceded_by_dollar:
            # Bare \gets - wrap it
            result.append(r'$\gets$')
            count += 1
            i += 5
        else:
            result.append(r'\gets')
            i += 5
    else:
        result.append(tex[i])
        i += 1

tex = ''.join(result)
TEX_FILE.write_text(tex)
print(f"Fixed {count} bare \\gets commands")
