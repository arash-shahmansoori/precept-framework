#!/usr/bin/env python3
"""
Convert algorithm pseudocode from lstlisting to native LaTeX algorithm environments.
Uses algorithm + algpseudocode packages.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEX_FILE = PROJECT_ROOT / "PRECEPT_PAPER.tex"


def escape_tex(s: str) -> str:
    """Escape special LaTeX chars in algorithm text, preserving existing LaTeX."""
    # Don't escape if already contains LaTeX commands
    if '\\' in s or '$' in s:
        return s
    s = s.replace('_', '\\_')
    s = s.replace('&', '\\&')
    s = s.replace('%', '\\%')
    return s


def format_code_id(s: str) -> str:
    """Format a code identifier for algorithm display."""
    # Wrap code identifiers in \texttt
    s = s.strip()
    if not s:
        return s
    return s


def parse_pseudocode_line(line: str, in_function: bool = False) -> str:
    """Convert a single pseudocode line to algpseudocode syntax."""
    stripped = line.strip()
    indent = len(line) - len(line.lstrip())

    # Skip empty lines
    if not stripped:
        return ''

    # Skip line numbers like "1:", "10:"
    stripped = re.sub(r'^\d+:\s*', '', stripped)
    if not stripped:
        return ''

    # Handle comments: // ... -> \Comment{...}
    if stripped.startswith('//'):
        comment_text = stripped[2:].strip()
        # Skip separator lines
        if all(c in '=- ' for c in comment_text) and len(comment_text) > 5:
            return ''
        return f'\\Comment{{{comment_text}}}'

    # Handle function definition
    m = re.match(r'function\s+(\w+)\(([^)]*)\):', stripped, re.IGNORECASE)
    if m:
        fname = m.group(1)
        fargs = m.group(2).strip()
        # Format args with $ for math-like params
        return f'\\Function{{{fname}}}{{{fargs}}}'

    # Handle end function
    if re.match(r'end\s+function', stripped, re.IGNORECASE):
        return '\\EndFunction'

    # Handle if-then
    m = re.match(r'if\s+(.+?)\s+then$', stripped, re.IGNORECASE)
    if m:
        cond = m.group(1)
        return f'\\If{{{cond}}}'

    # Handle else if / elif
    m = re.match(r'else\s+if\s+(.+?)\s+then$', stripped, re.IGNORECASE)
    if m:
        cond = m.group(1)
        return f'\\ElsIf{{{cond}}}'

    # Handle else
    if re.match(r'^else$', stripped, re.IGNORECASE):
        return '\\Else'

    # Handle end if
    if re.match(r'end\s+if', stripped, re.IGNORECASE):
        return '\\EndIf'

    # Handle for ... do
    m = re.match(r'for\s+(.+?)\s+do$', stripped, re.IGNORECASE)
    if m:
        loop_var = m.group(1)
        return f'\\For{{{loop_var}}}'

    # Handle end for
    if re.match(r'end\s+for', stripped, re.IGNORECASE):
        return '\\EndFor'

    # Handle while ... do
    m = re.match(r'while\s+(.+?)\s+do$', stripped, re.IGNORECASE)
    if m:
        cond = m.group(1)
        return f'\\While{{{cond}}}'

    # Handle end while
    if re.match(r'end\s+while', stripped, re.IGNORECASE):
        return '\\EndWhile'

    # Handle return
    m = re.match(r'return\s+(.*)', stripped, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return f'\\State \\Return {val}'

    # Handle try/finally blocks
    if stripped.lower().startswith('try:'):
        return '\\State \\textbf{try}'
    if stripped.lower().startswith('finally:'):
        return '\\State \\textbf{finally}'

    # Handle assignment with <-
    m = re.match(r'(.+?)\s*<-\s*(.+)', stripped)
    if m:
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        # Check for inline comment
        comment = ''
        cm = re.search(r'\s*//\s*(.+)$', rhs)
        if cm:
            comment = f' \\Comment{{{cm.group(1).strip()}}}'
            rhs = rhs[:cm.start()].strip()
        return f'\\State ${lhs} \\gets {rhs}${comment}'

    # Handle regular statements (add \State prefix)
    # Check for inline comment
    comment = ''
    text = stripped
    cm = re.search(r'\s*//\s*(.+)$', text)
    if cm:
        comment = f' \\Comment{{{cm.group(1).strip()}}}'
        text = text[:cm.start()].strip()

    return f'\\State {text}{comment}'


def extract_algorithm_parts(content: str):
    """Extract header info and body from an algorithm lstlisting block."""
    lines = content.split('\n')

    title = ''
    source = ''
    inputs = []
    outputs = []
    guards = []
    body_lines = []
    footer_lines = []

    in_body = False
    past_header = False
    in_footer = False

    for line in lines:
        stripped = line.strip()

        # Skip separator lines
        if re.match(r'^-{10,}$', stripped) or re.match(r'^={10,}$', stripped):
            if in_body and not in_footer:
                in_footer = True
            continue

        # Extract title
        m = re.match(r'Algorithm\s+[\d.]+[a-z]?:\s*(.+)', stripped)
        if m and not in_body:
            title = m.group(1).strip()
            continue

        # Extract source
        if stripped.startswith('Source:') and not in_body:
            source = stripped[7:].strip()
            continue
        # Continuation of source line
        if source and not in_body and not stripped.startswith('Input') and not stripped.startswith('Output') and not stripped.startswith('Guard') and not stripped.startswith('Constants') and not stripped.startswith('Global') and not stripped.startswith('function') and not re.match(r'^\d+:', stripped) and stripped and not stripped.startswith('//'):
            if not past_header:
                source += ' ' + stripped
                continue

        # Extract input
        if stripped.startswith('Input:') and not in_body:
            inputs.append(stripped[6:].strip())
            past_header = True
            continue

        # Extract output
        if stripped.startswith('Output:') and not in_body:
            outputs.append(stripped[7:].strip())
            past_header = True
            continue

        # Extract guards
        if stripped.startswith('Guards:') and not in_body:
            guards.append(stripped[7:].strip())
            continue

        # Constants and Global State as part of body
        if stripped.startswith('Constants:') or stripped.startswith('Global State:'):
            in_body = True
            past_header = True

        # Start of body (first function or statement)
        if not in_body and (stripped.startswith('function ') or
                           re.match(r'^\d+:\s*function', stripped) or
                           stripped.startswith('if ') or
                           stripped.startswith('// ') or
                           stripped.startswith('Constants') or
                           stripped.startswith('Global')):
            in_body = True
            past_header = True

        if in_footer:
            footer_lines.append(stripped)
        elif in_body:
            body_lines.append(line)

    return title, source, inputs, outputs, guards, body_lines, footer_lines


def convert_algorithm_block(content: str, alg_number: str) -> str:
    """Convert a full algorithm lstlisting block to algorithm environment."""
    title, source, inputs, outputs, guards, body_lines, footer_lines = extract_algorithm_parts(content)

    if not title:
        return None  # Not an algorithm block

    # Generate label from title
    label = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())[:30]
    label = re.sub(r'_+', '_', label).strip('_')

    # Build algorithm environment
    lines = []
    lines.append(f'\\begin{{algorithm}}[htbp]')
    lines.append(f'\\caption{{{title}}}')
    lines.append(f'\\label{{alg:{label}}}')

    # Add source as small text
    if source:
        lines.append(f'\\small\\textit{{Source: \\texttt{{{source}}}}}')

    lines.append('\\begin{algorithmic}[1]')

    # Input/Output
    if inputs:
        for inp in inputs:
            lines.append(f'\\Require {inp}')
    if outputs:
        for out in outputs:
            lines.append(f'\\Ensure {out}')

    # Process body
    for bline in body_lines:
        converted = parse_pseudocode_line(bline)
        if converted:
            lines.append(converted)

    lines.append('\\end{algorithmic}')

    # Footer (complexity, properties) as small text below
    if footer_lines:
        meaningful_footer = [f for f in footer_lines if f.strip() and not all(c in '-= ' for c in f.strip())]
        if meaningful_footer:
            lines.append('\\vspace{2pt}')
            lines.append('\\small')
            for fl in meaningful_footer:
                fl = fl.strip()
                if fl.startswith('Complexity:'):
                    lines.append(f'\\textit{{{fl}}}\\\\')
                elif fl.startswith('Property:'):
                    lines.append(f'\\textit{{{fl}}}')
                elif fl.startswith('Applicable:'):
                    lines.append(f'\\textit{{{fl}}}')
                else:
                    lines.append(f'\\textit{{{fl}}}\\\\')

    lines.append('\\end{algorithm}')

    return '\n'.join(lines)


def is_algorithm_block(content: str) -> bool:
    """Check if a lstlisting block contains algorithm pseudocode."""
    # Check for algorithm header pattern
    if re.search(r'Algorithm\s+[\d.]+[a-z]?:', content):
        return True
    # Check for function definition without being a Python code block
    if re.search(r'^function\s+\w+\(', content, re.MULTILINE) and 'def ' not in content and 'class ' not in content:
        return True
    return False


def process_file():
    """Process the full .tex file, converting algorithm lstlisting blocks."""
    tex = TEX_FILE.read_text(encoding='utf-8')

    # Find all lstlisting blocks
    pattern = r'(\\begin\{lstlisting\}(?:\[[^\]]*\])?)\n(.*?)\n(\\end\{lstlisting\})'
    matches = list(re.finditer(pattern, tex, re.DOTALL))

    print(f"Found {len(matches)} lstlisting blocks")

    # Process in reverse order to preserve positions
    converted = 0
    kept = 0

    for m in reversed(matches):
        begin_tag = m.group(1)
        content = m.group(2)
        end_tag = m.group(3)
        full_match = m.group(0)

        # Check if it has a language tag (Python code - keep as is)
        if '[language=' in begin_tag:
            kept += 1
            continue

        if is_algorithm_block(content):
            # Extract algorithm number from content
            alg_num_m = re.search(r'Algorithm\s+([\d.]+[a-z]?)', content)
            alg_num = alg_num_m.group(1) if alg_num_m else '?'

            result = convert_algorithm_block(content, alg_num)
            if result:
                # Also need to handle the preceding \textbf{Algorithm ...} label if present
                # Replace the lstlisting block
                start = m.start()
                end = m.end()

                tex = tex[:start] + result + tex[end:]
                converted += 1
                print(f"  [CONVERTED] Algorithm {alg_num}")
            else:
                kept += 1
                print(f"  [KEPT] Could not parse algorithm block")
        else:
            kept += 1

    TEX_FILE.write_text(tex, encoding='utf-8')
    print(f"\nConverted: {converted} algorithms")
    print(f"Kept as lstlisting: {kept} code snippets")


if __name__ == '__main__':
    process_file()
