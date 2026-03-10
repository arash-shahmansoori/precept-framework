#!/usr/bin/env python3
"""
Revert auto-converted algorithm environments to lstlisting wrapped in algorithm floats.
This gives us proper captions and numbering while avoiding complex escaping issues.
Keep the 3 hand-written algorithms (Epistemic Probe, Conflict Resolution, Thompson Sampling).
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEX_FILE = PROJECT_ROOT / "PRECEPT_PAPER.tex"
BACKUP = PROJECT_ROOT / "PRECEPT_PAPER_backup_before_algo.tex"

# Read the original pre-conversion version from git or the backup
# Since we don't have git, read the current and identify which algorithms to fix

# These are the hand-written algorithms that should be KEPT as algorithmic
KEEP_AS_ALGORITHMIC = {
    'alg:epistemic_probe',
    'alg:conflict_resolution', 
    'alg:thompson_sampling',
}

def main():
    tex = TEX_FILE.read_text()
    
    # Find all algorithm blocks
    pattern = re.compile(r'\\begin\{algorithm\}\[htbp\]\s*\n\\caption\{(.+?)\}\s*\n\\label\{(.+?)\}.*?\\end\{algorithm\}', re.DOTALL)
    
    matches = list(pattern.finditer(tex))
    print(f"Found {len(matches)} algorithm environments")
    
    converted = 0
    kept = 0
    
    for m in reversed(matches):
        caption = m.group(1)
        label = m.group(2)
        full = m.group(0)
        
        # Keep hand-written algorithms
        if label in KEEP_AS_ALGORITHMIC:
            kept += 1
            print(f"  [KEEP] {caption}")
            continue
        
        # Extract the algorithmic content and reconstruct pseudocode
        # Find the algorithmic block
        alg_match = re.search(r'\\begin\{algorithmic\}\[1\](.*?)\\end\{algorithmic\}', full, re.DOTALL)
        if not alg_match:
            kept += 1
            continue
        
        alg_content = alg_match.group(1)
        
        # Convert algorithmic commands back to pseudocode text
        pseudo = convert_algorithmic_to_pseudo(alg_content)
        
        # Extract source line if present
        source_match = re.search(r'\\small\\textit\{Source: \\texttt\{(.+?)\}\}', full)
        source = source_match.group(1) if source_match else ''
        
        # Extract footer
        footer_match = re.search(r'\\end\{algorithmic\}\s*\\vspace\{2pt\}\s*\\small\s*(.*?)\\end\{algorithm\}', full, re.DOTALL)
        footer_lines = []
        if footer_match:
            footer_text = footer_match.group(1)
            for fl in re.findall(r'\\textit\{(.+?)\}', footer_text):
                footer_lines.append(fl)
        
        # Build new algorithm with lstlisting
        new_lines = []
        new_lines.append(r'\begin{algorithm}[htbp]')
        new_lines.append(f'\\caption{{{caption}}}')
        new_lines.append(f'\\label{{{label}}}')
        new_lines.append(r'\begin{lstlisting}[frame=none,xleftmargin=1em,xrightmargin=0.5em]')
        if source:
            source_clean = source.replace('\\_', '_')
            new_lines.append(f'Source: {source_clean}')
            new_lines.append('')
        new_lines.append(pseudo)
        if footer_lines:
            new_lines.append('----------------------------------------------------------------------')
            for fl in footer_lines:
                fl_clean = fl.replace('\\texttt{', '').replace('}', '').replace('---', '--')
                new_lines.append(fl_clean)
        new_lines.append(r'\end{lstlisting}')
        new_lines.append(r'\end{algorithm}')
        
        replacement = '\n'.join(new_lines)
        tex = tex[:m.start()] + replacement + tex[m.end():]
        converted += 1
        print(f"  [REVERT] {caption}")
    
    TEX_FILE.write_text(tex)
    print(f"\nReverted: {converted}")
    print(f"Kept as algorithmic: {kept}")


def convert_algorithmic_to_pseudo(content):
    """Convert algorithmic commands back to clean pseudocode."""
    lines = content.strip().split('\n')
    result = []
    indent = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        prefix = '    ' * indent
        
        # Handle Require/Ensure
        if stripped.startswith('\\Require'):
            text = stripped.replace('\\Require', 'Input:').strip()
            result.append(text.replace('\\_', '_'))
            continue
        if stripped.startswith('\\Ensure'):
            text = stripped.replace('\\Ensure', 'Output:').strip()
            result.append(text.replace('\\_', '_'))
            result.append('')
            continue
        
        # Handle Function
        m = re.match(r'\\Function\{(\w+)\}\{(.+?)\}', stripped)
        if m:
            fname = m.group(1)
            fargs = m.group(2)
            result.append(f'{prefix}function {fname}({fargs}):'.replace('\\_', '_'))
            indent += 1
            continue
        
        if stripped == '\\EndFunction':
            indent = max(0, indent - 1)
            result.append(f'{"    " * indent}end function')
            continue
        
        # Handle If
        m = re.match(r'\\If\{(.+)\}', stripped)
        if m:
            cond = m.group(1)
            result.append(f'{prefix}if {cond} then'.replace('\\_', '_').replace('$', ''))
            indent += 1
            continue
        
        m = re.match(r'\\ElsIf\{(.+)\}', stripped)
        if m:
            indent = max(0, indent - 1)
            cond = m.group(1)
            result.append(f'{"    " * indent}else if {cond} then'.replace('\\_', '_').replace('$', ''))
            indent += 1
            continue
        
        if stripped == '\\Else':
            indent = max(0, indent - 1)
            result.append(f'{"    " * indent}else')
            indent += 1
            continue
        
        if stripped == '\\EndIf':
            indent = max(0, indent - 1)
            result.append(f'{"    " * indent}end if')
            continue
        
        # Handle For
        m = re.match(r'\\For\{(.+)\}', stripped)
        if m:
            loop = m.group(1)
            result.append(f'{prefix}for {loop} do'.replace('\\_', '_').replace('$', ''))
            indent += 1
            continue
        
        if stripped == '\\EndFor':
            indent = max(0, indent - 1)
            result.append(f'{"    " * indent}end for')
            continue
        
        # Handle While
        m = re.match(r'\\While\{(.+)\}', stripped)
        if m:
            cond = m.group(1)
            result.append(f'{prefix}while {cond} do'.replace('\\_', '_').replace('$', ''))
            indent += 1
            continue
        
        if stripped == '\\EndWhile':
            indent = max(0, indent - 1)
            result.append(f'{"    " * indent}end while')
            continue
        
        # Handle State with Return
        if '\\Return' in stripped:
            text = stripped.replace('\\State', '').replace('\\Return', 'return').strip()
            text = clean_latex(text)
            result.append(f'{prefix}{text}')
            continue
        
        # Handle Comment
        m = re.match(r'\\Comment\{(.+)\}', stripped)
        if m:
            result.append(f'{prefix}// {m.group(1)}'.replace('\\_', '_'))
            continue
        
        # Handle Statex + Comment
        m = re.match(r'\\Statex\s*\\Comment\{(.+)\}', stripped)
        if m:
            result.append(f'{prefix}// {m.group(1)}'.replace('\\_', '_'))
            continue
        
        if stripped == '\\Statex':
            result.append('')
            continue
        
        # Handle LineComment
        m = re.match(r'\\LineComment\{(.+)\}', stripped)
        if m:
            result.append(f'{prefix}// {m.group(1)}'.replace('\\_', '_'))
            continue
        
        # Handle State
        if stripped.startswith('\\State'):
            text = stripped[6:].strip()
            text = clean_latex(text)
            result.append(f'{prefix}{text}')
            continue
        
        # Fallback
        text = clean_latex(stripped)
        result.append(f'{prefix}{text}')
    
    return '\n'.join(result)


def clean_latex(text):
    """Remove LaTeX formatting from algorithmic text."""
    text = text.replace('\\_', '_')
    text = text.replace('$\\gets$', '<-')
    text = text.replace('\\gets', '<-')
    text = text.replace('$', '')
    text = text.replace('\\textbf{', '').replace('\\texttt{', '').replace('\\textit{', '')
    # Remove closing braces that were from \textbf{} etc
    # Be careful not to remove structural braces
    text = re.sub(r'\\Comment\{(.+?)\}', r'// \1', text)
    return text


if __name__ == '__main__':
    main()
