#!/usr/bin/env python3
"""
Fix escaping issues in converted algorithm environments.
Inside algorithmic environments, underscores must be escaped as \_ in text mode,
but NOT inside $...$ math blocks.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEX_FILE = PROJECT_ROOT / "PRECEPT_PAPER.tex"


def fix_underscores_in_text(line: str) -> str:
    """Escape underscores in text mode, preserving math mode blocks."""
    if not line.strip():
        return line
    
    # Protect existing $...$ blocks
    saved = []
    counter = [0]
    
    def save(m):
        saved.append(m.group(0))
        idx = counter[0]
        counter[0] += 1
        return f'XMATH{idx}XMATH'
    
    # Save math blocks
    result = re.sub(r'\$[^$]+\$', save, line)
    
    # Save \texttt{} blocks - convert underscores inside them
    def save_texttt(m):
        inner = m.group(1)
        inner = inner.replace('_', '\\_')
        saved.append(f'\\texttt{{{inner}}}')
        idx = counter[0]
        counter[0] += 1
        return f'XMATH{idx}XMATH'
    result = re.sub(r'\\texttt\{([^}]*)\}', save_texttt, result)
    
    # Now escape remaining underscores (that aren't already escaped)
    result = re.sub(r'(?<!\\)_', r'\\_', result)
    
    # Restore saved blocks
    for i, block in enumerate(saved):
        result = result.replace(f'XMATH{i}XMATH', block)
    
    return result


def fix_algorithm_block(content: str) -> str:
    """Fix a complete algorithm environment block."""
    lines = content.split('\n')
    fixed = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip lines that are pure LaTeX structural commands
        if stripped.startswith('\\begin{') or stripped.startswith('\\end{'):
            fixed.append(line)
            continue
        if stripped.startswith('\\caption{') or stripped.startswith('\\label{'):
            fixed.append(line)
            continue
        if stripped.startswith('\\small\\textit{Source:'):
            # Fix underscores in source references
            fixed.append(fix_underscores_in_text(line))
            continue
        if stripped.startswith('\\vspace') or stripped == '\\small':
            fixed.append(line)
            continue
            
        # For algorithmic lines, fix underscores
        if any(stripped.startswith(cmd) for cmd in [
            '\\Function', '\\EndFunction', '\\Procedure', '\\EndProcedure',
            '\\If', '\\ElsIf', '\\Else', '\\EndIf',
            '\\For', '\\EndFor', '\\While', '\\EndWhile',
            '\\State', '\\Return', '\\Require', '\\Ensure',
            '\\Comment', '\\Statex', '\\LineComment',
            '\\textit{'
        ]):
            fixed.append(fix_underscores_in_text(line))
        else:
            fixed.append(fix_underscores_in_text(line))
    
    return '\n'.join(fixed)


def main():
    tex = TEX_FILE.read_text(encoding='utf-8')
    
    # Find all algorithm blocks
    pattern = r'(\\begin\{algorithm\}.*?\\end\{algorithm\})'
    matches = list(re.finditer(pattern, tex, re.DOTALL))
    
    print(f"Found {len(matches)} algorithm environments")
    
    # Process in reverse order
    for i, m in enumerate(reversed(matches)):
        alg_content = m.group(0)
        fixed = fix_algorithm_block(alg_content)
        
        if fixed != alg_content:
            tex = tex[:m.start()] + fixed + tex[m.end():]
            # Extract caption for logging
            cap_m = re.search(r'\\caption\{(.+?)\}', alg_content)
            name = cap_m.group(1) if cap_m else f"algorithm {i}"
            print(f"  Fixed: {name}")
    
    TEX_FILE.write_text(tex, encoding='utf-8')
    
    # Verify: count unescaped underscores in algorithm environments
    tex = TEX_FILE.read_text(encoding='utf-8')
    alg_blocks = re.findall(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', tex, re.DOTALL)
    
    issues = 0
    for block in alg_blocks:
        # Check for unescaped underscores (not in math mode)
        # Remove math blocks first
        clean = re.sub(r'\$[^$]+\$', '', block)
        # Count unescaped underscores
        bare = re.findall(r'(?<!\\)_', clean)
        issues += len(bare)
    
    print(f"\nRemaining unescaped underscores in algorithmic: {issues}")


if __name__ == '__main__':
    main()
