#!/usr/bin/env python3
"""
Comprehensive formatting fix for PRECEPT_PAPER.tex:
1. Fix figure sizing - reduce whitespace, use max height
2. Fix ALL tables - wrap in resizebox
3. Fix lstlisting - proper underscore handling, reduce font, limit width
4. Fix float spacing
5. Trim Mermaid figure PDFs to remove whitespace
"""

import re
import subprocess
from pathlib import Path

TEX_PATH = Path(__file__).parent.parent / "PRECEPT_PAPER.tex"
MERMAID_DIR = Path(__file__).parent.parent / "figures" / "mermaid"


def trim_pdfs():
    """Use pdfcrop to trim whitespace from all Mermaid PDFs."""
    print("=== Trimming PDF whitespace ===")
    for pdf in sorted(MERMAID_DIR.glob("*.pdf")):
        try:
            result = subprocess.run(
                ["pdfcrop", str(pdf), str(pdf)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"  [OK] {pdf.name}")
            else:
                print(f"  [WARN] {pdf.name}: {result.stderr[:100]}")
        except FileNotFoundError:
            print("  [SKIP] pdfcrop not found - install texlive-extra-utils")
            return False
        except Exception as e:
            print(f"  [ERR] {pdf.name}: {e}")
    return True


def fix_preamble(lines):
    """Fix preamble: lstlisting settings, float spacing, figure sizing."""
    result = []
    for i, line in enumerate(lines):
        # Replace lstset block entirely
        if line.strip() == r'\lstset{':
            result.append(r'\lstset{')
            result.append(r'  basicstyle=\ttfamily\footnotesize,')
            result.append(r'  breaklines=true,')
            result.append(r'  breakatwhitespace=false,')
            result.append(r'  frame=single,')
            result.append(r'  numbers=none,')
            result.append(r'  backgroundcolor=\color{gray!5},')
            result.append(r'  showstringspaces=false,')
            result.append(r'  tabsize=2,')
            result.append(r'  xleftmargin=0.5em,')
            result.append(r'  xrightmargin=0.5em,')
            result.append(r'  columns=fullflexible,')
            result.append(r'  keepspaces=true,')
            result.append(r'  escapeinside={(*@}{@*)},')
            result.append(r'}')
            # Skip old lstset content
            j = i + 1
            while j < len(lines) and lines[j].strip() != '}':
                j += 1
            # Skip closing }
            i = j + 1
            # Continue from after the old lstset
            while i < len(lines):
                result.append(lines[i])
                i += 1
            return result

        result.append(line)

    return result


def fix_figures(lines):
    """Fix figure sizing: add max height, reduce widths."""
    result = []
    for i, line in enumerate(lines):
        # Fix figure includes - add max height constraint
        if r'\includegraphics[width=' in line and 'mermaid/' in line:
            # Extract current width
            m = re.match(r'(\s*)\\includegraphics\[width=([^\]]+)\]\{(.+)\}', line)
            if m:
                indent = m.group(1)
                width = m.group(2)
                path = m.group(3)

                # Determine appropriate sizing based on diagram type
                if 'execution_flow' in path:
                    # Very tall diagram - limit height more
                    new_line = f'{indent}\\includegraphics[width=0.92\\textwidth,height=0.75\\textheight,keepaspectratio]{{{path}}}'
                elif 'compass_architecture' in path or 'simplified_pipeline' in path:
                    new_line = f'{indent}\\includegraphics[width=0.9\\textwidth,height=0.65\\textheight,keepaspectratio]{{{path}}}'
                elif 'tier_hierarchy' in path:
                    # Small/simple diagram
                    new_line = f'{indent}\\includegraphics[width=0.5\\textwidth,height=0.2\\textheight,keepaspectratio]{{{path}}}'
                elif 'positioning_summary' in path or 'ablation' in path:
                    new_line = f'{indent}\\includegraphics[width=0.8\\textwidth,height=0.55\\textheight,keepaspectratio]{{{path}}}'
                elif any(x in path for x in ['compositional', 'drift', 'learning_curves']):
                    # Chart diagrams
                    new_line = f'{indent}\\includegraphics[width=0.75\\textwidth,height=0.5\\textheight,keepaspectratio]{{{path}}}'
                else:
                    # Default - constrain both width and height
                    new_line = f'{indent}\\includegraphics[width=0.85\\textwidth,height=0.6\\textheight,keepaspectratio]{{{path}}}'

                result.append(new_line)
                continue

        result.append(line)

    return result


def fix_tables(lines):
    """Wrap ALL tables in resizebox to prevent overflow."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect \begin{tabular}
        if r'\begin{tabular}' in line.strip():
            # Check if already wrapped in resizebox
            already_wrapped = False
            if len(result) > 0 and 'resizebox' in result[-1]:
                already_wrapped = True

            if not already_wrapped:
                # Insert resizebox wrapper
                result.append(r'\resizebox{\textwidth}{!}{%')
                result.append(line)
                i += 1
                # Find matching \end{tabular}
                depth = 1
                while i < len(lines):
                    result.append(lines[i])
                    if r'\begin{tabular}' in lines[i]:
                        depth += 1
                    if r'\end{tabular}' in lines[i]:
                        depth -= 1
                        if depth == 0:
                            result.append('}%')  # Close resizebox
                            i += 1
                            break
                    i += 1
                continue
            else:
                result.append(line)
                i += 1
                continue

        result.append(line)
        i += 1

    return result


def fix_float_spacing(lines):
    """Add float spacing settings after \\begin{document}."""
    result = []
    for line in lines:
        result.append(line)
        if line.strip() == r'\begin{document}':
            result.append('')
            result.append(r'% ── Float spacing ──')
            result.append(r'\setlength{\floatsep}{8pt plus 2pt minus 2pt}')
            result.append(r'\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}')
            result.append(r'\setlength{\intextsep}{8pt plus 2pt minus 2pt}')
            result.append(r'\setlength{\abovecaptionskip}{5pt}')
            result.append(r'\setlength{\belowcaptionskip}{3pt}')
            result.append('')
    return result


def fix_code_blocks(lines):
    """Fix code blocks: remove box-drawing chars rendered as dashes, reduce separator width."""
    result = []
    in_lstlisting = False
    for line in lines:
        if r'\begin{lstlisting}' in line:
            in_lstlisting = True
            result.append(line)
            continue
        if r'\end{lstlisting}' in line:
            in_lstlisting = False
            result.append(line)
            continue

        if in_lstlisting:
            # Replace long separator lines with shorter ones
            if re.match(r'^[-=]{40,}$', line.strip()):
                line = '-' * 70
            # Truncate very long lines in code blocks
            if len(line) > 90:
                # Try to break at a reasonable point
                line = line[:88] + ' ...'
            result.append(line)
        else:
            result.append(line)

    return result


def main():
    print("=" * 60)
    print("COMPREHENSIVE FORMATTING FIX")
    print("=" * 60)

    # Step 1: Trim PDFs
    trim_pdfs()

    # Step 2: Read .tex file
    with open(TEX_PATH, 'r') as f:
        lines = f.read().split('\n')

    print(f"\nLoaded {len(lines)} lines from {TEX_PATH}")

    # Step 3: Apply fixes in order
    print("\n1. Fixing preamble (lstset)...")
    lines = fix_preamble(lines)

    print("2. Fixing float spacing...")
    lines = fix_float_spacing(lines)

    print("3. Fixing figure sizing...")
    lines = fix_figures(lines)

    print("4. Fixing tables...")
    lines = fix_tables(lines)

    print("5. Fixing code blocks...")
    lines = fix_code_blocks(lines)

    # Step 4: Write result
    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nWrote {len(lines)} lines to {TEX_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
