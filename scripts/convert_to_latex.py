#!/usr/bin/env python3
"""
Convert PRECEPT_PAPER.md to LaTeX format suitable for JAIR / TMLR submission.

Handles:
- Section hierarchy (##/###/####/##### → section/subsection/subsubsection/paragraph)
- Markdown tables → LaTeX tabular
- Citations → \citep{} with auto-generated BibTeX
- References → BibTeX entries
- Mermaid diagrams → figure placeholders
- Code blocks → lstlisting
- Unicode math → LaTeX commands
- Bold/Italic → \textbf{}/\textit{}
- Lists → itemize/enumerate
- Images → figure environments
- Special character escaping
"""

import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# UNICODE → LATEX MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

UNICODE_MAP = {
    'α': r'$\alpha$',
    'β': r'$\beta$',
    'γ': r'$\gamma$',
    'δ': r'$\delta$',
    'ε': r'$\varepsilon$',
    'θ': r'$\theta$',
    'λ': r'$\lambda$',
    'μ': r'$\mu$',
    'π': r'$\pi$',
    'σ': r'$\sigma$',
    'τ': r'$\tau$',
    'φ': r'$\varphi$',
    'Δ': r'$\Delta$',
    'Σ': r'$\Sigma$',
    'Ω': r'$\Omega$',
    '×': r'$\times$',
    '→': r'$\rightarrow$',
    '←': r'$\leftarrow$',
    '↔': r'$\leftrightarrow$',
    '⟶': r'$\longrightarrow$',
    '≥': r'$\geq$',
    '≤': r'$\leq$',
    '≠': r'$\neq$',
    '≈': r'$\approx$',
    '∈': r'$\in$',
    '∉': r'$\notin$',
    '⊂': r'$\subset$',
    '∪': r'$\cup$',
    '∩': r'$\cap$',
    '∅': r'$\emptyset$',
    '∞': r'$\infty$',
    '∑': r'$\sum$',
    '∏': r'$\prod$',
    '√': r'$\sqrt{}$',
    '±': r'$\pm$',
    '·': r'$\cdot$',
    '°': r'$^\circ$',
    '²': r'$^2$',
    '³': r'$^3$',
    '¹': r'$^1$',
    '₁': r'$_1$',
    '₂': r'$_2$',
    '₃': r'$_3$',
    '₀': r'$_0$',
    '✓': r'\checkmark{}',
    '✗': r'$\times$',
    '—': '---',
    '–': '--',
    '"': "``",
    '"': "''",
    ''': "`",
    ''': "'",
    '…': r'\ldots{}',
    '≻': r'$\succ$',
    '∀': r'$\forall$',
    '∃': r'$\exists$',
    '□': r'$\square$',
    '▪': r'$\blacksquare$',
    'κ': r'$\kappa$',
    '❌': r'$\times$',
    '∎': r'$\blacksquare$',
    '𝒫': r'$\mathcal{P}$',
    '⊆': r'$\subseteq$',
    '⊇': r'$\supseteq$',
    'ₖ': r'$_k$',
    'ₘ': r'$_m$',
    'ₚ': r'$_p$',
    'ℓ': r'$\ell$',
    '∧': r'$\wedge$',
    '∨': r'$\vee$',
    '¬': r'$\neg$',
    '⊕': r'$\oplus$',
    '⊗': r'$\otimes$',
    '≡': r'$\equiv$',
    '⇒': r'$\Rightarrow$',
    '⇐': r'$\Leftarrow$',
    '⇔': r'$\Leftrightarrow$',
    '−': '-',  # Unicode minus → ASCII minus
    '§': r'\S{}',  # Section sign
    '△': r'$\triangle$',
    'ρ': r'$\rho$',
    '~': r'$\sim$',
}

# Subscript letter mappings for P₁, Pₜ etc.
SUBSCRIPT_LETTERS = {
    '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4', '₅': '_5',
    '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9', '₀': '_0',
    'ₐ': '_a', 'ₑ': '_e', 'ₒ': '_o', 'ₓ': '_x',
    'ₜ': '_t', 'ₙ': '_n', 'ᵢ': '_i', 'ⱼ': '_j',
    'ₖ': '_k', 'ₘ': '_m', 'ₚ': '_p',
}

# Unicode → ASCII for code blocks (lstlisting can't handle Unicode with pdflatex)
CODE_SANITIZE = {
    '━': '-', '═': '=', '║': '|', '│': '|', '─': '-',
    '┌': '+', '┐': '+', '└': '+', '┘': '+', '├': '+', '┤': '+',
    '┬': '+', '┴': '+', '┼': '+', '╔': '+', '╗': '+', '╚': '+',
    '╝': '+', '╠': '+', '╣': '+', '╦': '+', '╩': '+', '╬': '+',
    '←': '<-', '→': '->', '↔': '<->', '⟶': '-->', '⇒': '=>',
    '≥': '>=', '≤': '<=', '≠': '!=', '≈': '~=', '∈': 'in',
    '∉': 'not in', '⊂': 'subset', '∪': 'union', '∩': 'intersect',
    '∅': 'empty', '∞': 'inf', '×': 'x', '±': '+/-', '·': '.',
    '°': 'deg', '…': '...', '—': '---', '–': '--',
    '"': '"', '"': '"', ''': "'", ''': "'",
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu',
    'π': 'pi', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'κ': 'kappa',
    'Δ': 'Delta', 'Σ': 'Sigma', 'Ω': 'Omega',
    '₁': '_1', '₂': '_2', '₃': '_3', '₀': '_0', 'ₜ': '_t',
    'ₙ': '_n', 'ᵢ': '_i', 'ⱼ': '_j',
    '✓': '[Y]', '✗': '[N]',
    '≻': '>',
    '∀': 'forall', '∃': 'exists',
    '□': '[]', '▪': '[*]',
    '²': '^2', '³': '^3', '¹': '^1',
    '−': '-', '§': 'S', '△': '^',
    'ρ': 'rho',
}


def sanitize_code_line(line: str) -> str:
    """Replace Unicode characters with ASCII equivalents for lstlisting."""
    for char, replacement in CODE_SANITIZE.items():
        line = line.replace(char, replacement)
    # Remove any remaining non-ASCII characters
    result = []
    for ch in line:
        if ord(ch) < 128:
            result.append(ch)
        else:
            result.append('?')
    return ''.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CITATION KEY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cite_key(author_str: str, year: str) -> str:
    """Generate a BibTeX citation key from author and year."""
    # Get first author's last name
    author_str = author_str.strip()
    # Handle "Last, F." or "Last, First" format
    if ',' in author_str.split('&')[0].split(' et al')[0]:
        last_name = author_str.split(',')[0].strip()
    else:
        parts = author_str.split()
        last_name = parts[0] if parts else 'unknown'

    # Clean the name
    last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    return f"{last_name}{year}"


def parse_references(lines: List[str], ref_start: int) -> Tuple[Dict[str, str], str]:
    """Parse the References section into BibTeX entries and a citation key map."""
    bibtex_entries = []
    cite_map = {}  # Maps "(Author et al., 2024)" style to cite key

    i = ref_start
    while i < len(lines):
        line = lines[i].strip()

        # Stop at next major section (Appendix, ---)
        if line.startswith('## Appendix') or (line == '---' and i > ref_start + 2):
            break

        if not line or line.startswith('## '):
            i += 1
            continue

        # Parse reference line: "Author(s) (YEAR). Title. *Venue*."
        ref_match = re.match(
            r'^(.+?)\s*\((\d{4})\)\.\s*(.+?)(?:\.\s*\*(.+?)\*)?\.?\s*$',
            line
        )
        if ref_match:
            authors = ref_match.group(1).strip()
            year = ref_match.group(2)
            title = ref_match.group(3).strip().rstrip('.')
            venue = ref_match.group(4) or ''
            venue = venue.strip().rstrip('.')

            key = generate_cite_key(authors, year)
            # Handle duplicate keys
            base_key = key
            suffix = 'b'
            while key in cite_map.values():
                key = base_key + suffix
                suffix = chr(ord(suffix) + 1)

            # Determine entry type
            if 'arXiv' in venue:
                entry_type = 'article'
                arxiv_id = re.search(r'arXiv:(\S+)', venue)
                venue_field = f"  journal = {{arXiv preprint}},\n"
                if arxiv_id:
                    venue_field += f"  eprint = {{{arxiv_id.group(1)}}},\n"
                    venue_field += f"  archivePrefix = {{arXiv}},\n"
            elif venue:
                entry_type = 'inproceedings'
                venue_field = f"  booktitle = {{{venue}}},\n"
            else:
                entry_type = 'article'
                venue_field = ''

            # Format authors for BibTeX
            bib_authors = authors.replace(' & ', ' and ').replace(', et al.', ' and others').replace(' et al.', ' and others')

            entry = (
                f"@{entry_type}{{{key},\n"
                f"  author = {{{bib_authors}}},\n"
                f"  title = {{{title}}},\n"
                f"{venue_field}"
                f"  year = {{{year}}},\n"
                f"}}\n"
            )
            bibtex_entries.append(entry)

            # Build citation map entries for various in-text citation forms
            first_author = authors.split(',')[0].strip().split()[-1] if ',' in authors else authors.split()[0]

            if 'et al' in authors or '&' in authors:
                cite_map[f"({first_author} et al., {year})"] = key
                cite_map[f"{first_author} et al., {year}"] = key
                cite_map[f"{first_author} et al. ({year})"] = key
                cite_map[f"({first_author} et al., {year};"] = key
            else:
                cite_map[f"({first_author}, {year})"] = key
                cite_map[f"{first_author}, {year}"] = key
                cite_map[f"{first_author} ({year})"] = key

            # Also handle "Author & Author" citations
            if '&' in authors and 'et al' not in authors:
                authors_short = authors.split('(')[0].strip()
                cite_map[f"({authors_short}, {year})"] = key

        i += 1

    bib_content = '\n'.join(bibtex_entries)
    return cite_map, bib_content


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN TABLE → LATEX
# ═══════════════════════════════════════════════════════════════════════════════

def convert_table(table_lines: List[str]) -> str:
    """Convert markdown table lines to LaTeX tabular."""
    if not table_lines:
        return ''

    # Parse header
    header = table_lines[0]
    cells = [c.strip() for c in header.split('|')[1:-1]]
    n_cols = len(cells)

    # Determine alignment from separator line
    if len(table_lines) > 1:
        sep = table_lines[1]
        sep_cells = [c.strip() for c in sep.split('|')[1:-1]]
        aligns = []
        for sc in sep_cells:
            sc = sc.strip('-').strip()
            if sc.startswith(':') and sc.endswith(':'):
                aligns.append('c')
            elif sc.endswith(':'):
                aligns.append('r')
            else:
                aligns.append('l')
        # Pad if needed
        while len(aligns) < n_cols:
            aligns.append('l')
    else:
        aligns = ['l'] * n_cols

    col_spec = '|'.join(aligns)

    # Build LaTeX
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\small')
    latex_lines.append(r'\begin{tabular}{' + col_spec + '}')
    latex_lines.append(r'\toprule')

    # Header row
    header_cells = [convert_inline(c.strip()) for c in table_lines[0].split('|')[1:-1]]
    latex_lines.append(' & '.join(header_cells) + r' \\')
    latex_lines.append(r'\midrule')

    # Data rows (skip separator at index 1)
    for row_line in table_lines[2:]:
        if not row_line.strip():
            continue
        row_cells = [c.strip() for c in row_line.split('|')[1:-1]]
        # Pad if needed
        while len(row_cells) < n_cols:
            row_cells.append('')
        converted_cells = [convert_inline(c) for c in row_cells]
        latex_lines.append(' & '.join(converted_cells) + r' \\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    return '\n'.join(latex_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE FORMATTING CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def convert_inline(text: str) -> str:
    """Convert inline markdown formatting to LaTeX."""
    if not text:
        return text

    # ── Phase 1: Save protected regions ──
    saved = []
    counter = [0]

    def save_block(m):
        saved.append(m.group(0))
        idx = counter[0]
        counter[0] += 1
        return f'XSAVED{idx}ENDX'

    # Save display math $$...$$ (must come before inline $...$)
    text = re.sub(r'\$\$.+?\$\$', save_block, text, flags=re.DOTALL)
    # Save inline math $...$
    text = re.sub(r'\$.+?\$', save_block, text)

    # ── Phase 2: Convert markdown formatting (before escaping) ──

    # Convert bold+italic ***text*** -> \textbf{\textit{text}}
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\\textbf{\\textit{\1}}', text)
    # Convert bold **text** -> \textbf{text}
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # Convert italic *text* -> \textit{text}
    text = re.sub(r'(?<![\\w*])\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)

    # Convert inline code `text` -> \texttt{text}  (escape specials inside)
    def convert_code(m):
        code = m.group(1)
        code = code.replace('_', r'\_')
        code = code.replace('&', r'\&')
        code = code.replace('%', r'\%')
        # Replace Greek letters with ascii in texttt
        code = code.replace('κ', r'$\kappa$')
        code = code.replace('α', r'$\alpha$')
        code = code.replace('β', r'$\beta$')
        code = code.replace('θ', r'$\theta$')
        return f'\\texttt{{{code}}}'
    text = re.sub(r'`([^`]+)`', convert_code, text)

    # Convert markdown links [text](url) -> \href{url}{text}
    def convert_link(m):
        link_text = m.group(1)
        url = m.group(2)
        url = url.replace('_', r'\_')
        url = url.replace('%', r'\%')
        return f'\\href{{{url}}}{{{link_text}}}'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', convert_link, text)

    # ── Phase 3: Detect and save bare math expressions ──
    # These need to be saved BEFORE Unicode conversion to avoid nested $...$

    def math_unicode(s):
        """Convert Unicode inside a math expression to LaTeX commands."""
        s = s.replace('×', r'\times')
        s = s.replace('β', r'\beta')
        s = s.replace('α', r'\alpha')
        s = s.replace('θ', r'\theta')
        s = s.replace('κ', r'\kappa')
        s = s.replace('≥', r'\geq')
        s = s.replace('≤', r'\leq')
        s = s.replace('≈', r'\approx')
        s = s.replace('·', r'\cdot')
        return s

    def wrap_math_expr(m):
        inner = math_unicode(m.group(0))
        saved.append(f'${inner}$')
        idx = counter[0]
        counter[0] += 1
        return f'XSAVED{idx}ENDX'

    # O(...) expressions — big-O notation
    text = re.sub(r'(?<!\$)\bO\([^)]+\)', wrap_math_expr, text)
    # 2^N, 2^6, 2^{N-1} etc.
    text = re.sub(r'(?<!\$)\b(\d+)\^([A-Za-z0-9]+(?:[-+][A-Za-z0-9]+)*)', wrap_math_expr, text)
    # p^N, p^{10} etc.
    text = re.sub(r'(?<!\$)\b([a-z])\^(\d+|[A-Za-z])', wrap_math_expr, text)
    # N=10 style
    text = re.sub(r'(?<!\$)(?<![\\a-zA-Z])([A-Z])\s*=\s*(\d+)', lambda m: f'${m.group(1)}={m.group(2)}$', text)

    # ── Phase 4: Convert Unicode ──
    for char, latex in UNICODE_MAP.items():
        text = text.replace(char, latex)

    # Handle letter+subscript patterns (P₁, Pₜ)
    for sub_char, sub_latex in SUBSCRIPT_LETTERS.items():
        text = re.sub(
            r'([A-Za-z])' + re.escape(sub_char),
            lambda m: f'${m.group(1)}{sub_latex}$',
            text
        )

    # ── Phase 4: Escape special LaTeX characters ──
    # Save \texttt{}, \textbf{}, \textit{}, \href{}, \citep{}, \citet{},
    # and other LaTeX commands so we don't double-escape their content
    latex_cmds = []
    lcounter = [0]

    def save_latex_cmd(m):
        latex_cmds.append(m.group(0))
        idx = lcounter[0]
        lcounter[0] += 1
        return f'XLATEX{idx}ENDX'

    # Save all \command{...} sequences (handles nested braces one level deep)
    text = re.sub(
        r'\\(?:texttt|textbf|textit|textbf\{\\textit|href|citep|citet|checkmark|ldots|sqrt)\{(?:[^{}]|\{[^{}]*\})*\}',
        save_latex_cmd, text
    )

    # Now escape special characters in the remaining plain text
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    # Escape underscores NOT already escaped
    text = re.sub(r'(?<!\\)_', r'\\_', text)

    # Restore LaTeX commands
    for idx, cmd in enumerate(latex_cmds):
        text = text.replace(f'XLATEX{idx}ENDX', cmd)

    # ── Phase 6: Catch any remaining non-ASCII ──
    # Replace any Unicode that slipped through
    remaining_unicode = {
        'κ': r'$\kappa$', 'ᵢ': r'$_i$', 'ⱼ': r'$_j$',
        '❌': r'$\times$', '∎': r'$\blacksquare$',
        '𝒫': r'$\mathcal{P}$', '⊆': r'$\subseteq$',
    }
    for char, repl in remaining_unicode.items():
        text = text.replace(char, repl)

    # ── Phase 7: Restore saved math blocks ──
    for idx, block in enumerate(saved):
        text = text.replace(f'XSAVED{idx}ENDX', block)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# CITATION REPLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess_latex(lines: List[str]) -> List[str]:
    """Fix common LaTeX issues in the final output."""
    result = []
    for line in lines:
        # Replace any remaining Unicode minus
        line = line.replace('\u2212', '-')
        # Replace any remaining section sign
        line = line.replace('\u00a7', r'\S{}')
        # Replace any remaining triangle
        line = line.replace('\u25b3', r'$\triangle$')
        # Replace Unicode superscript 2
        line = line.replace('\u00b2', r'$^2$')
        # Replace rho
        line = line.replace('\u03c1', r'$\rho$')
        # Replace subscript i, 1
        line = line.replace('\u1d62', r'_i')
        line = line.replace('\u2081', r'_1')

        # Fix unescaped % inside \textbf{}, \textit{}, etc.
        # Apply repeatedly to catch all instances
        for _ in range(3):
            line = re.sub(
                r'(\\text(?:bf|it|tt)\{[^}]*)(?<!\\)%([^}]*\})',
                lambda m: m.group(1) + r'\%' + m.group(2),
                line
            )
        # Fix bare & in text (not in tabular column separators)
        # Only fix & that are not column separators (preceded by whitespace, not \ )
        # Actually, & in table cells are column separators, leave them alone

        # Fix $\times$ adjacent to numbers: "64$\times$" → "$64\times$"
        line = re.sub(r'(\d+)\$\\times\$', r'$\1\\times$', line)

        # Fix bare _ in text (not in math, not already escaped)
        # This is a safety net for any that slipped through
        # First, save math regions
        parts = re.split(r'(\$[^$]+\$|\$\$[^$]+\$\$)', line)
        fixed_parts = []
        for j, part in enumerate(parts):
            if j % 2 == 0:  # Not in math
                # Don't touch \_ (already escaped) or _ in lstlisting
                part = re.sub(r'(?<!\\)_(?![{])', r'\\_', part)
            fixed_parts.append(part)
        line = ''.join(fixed_parts)

        result.append(line)
    return result


def replace_citations(text: str, cite_map: Dict[str, str]) -> str:
    """Replace in-text citations with \\citep{} commands."""
    # Sort by length (longest first) to avoid partial matches
    for cite_text, key in sorted(cite_map.items(), key=lambda x: -len(x[0])):
        if cite_text in text:
            # Determine if parenthetical or textual citation
            if cite_text.startswith('('):
                text = text.replace(cite_text, f'\\citep{{{key}}}')
            else:
                text = text.replace(cite_text, f'\\citet{{{key}}}')
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONVERTER
# ═══════════════════════════════════════════════════════════════════════════════

def convert_md_to_latex(md_path: str, tex_path: str, bib_path: str):
    """Main conversion function."""
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Strip trailing newlines
    lines = [l.rstrip('\n') for l in lines]

    # ── Step 1: Find references section and parse ──
    ref_start = None
    for i, line in enumerate(lines):
        if line.strip() == '## References':
            ref_start = i
            break

    cite_map = {}
    bib_content = ''
    if ref_start is not None:
        cite_map, bib_content = parse_references(lines, ref_start)

    # Write BibTeX file
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(bib_content)
    print(f"  Generated {bib_path} with {bib_content.count('@')} entries")

    # ── Step 2: Build LaTeX document ──
    output = []

    # Document preamble
    output.append(generate_preamble())

    # Process content
    i = 0
    in_code_block = False
    code_lang = ''
    code_lines = []
    in_table = False
    table_lines = []
    in_list = False
    list_type = None  # 'itemize' or 'enumerate'
    list_items = []
    skip_until_section = False  # Skip references section
    title_seen = False  # Track whether we've passed the # title line
    skip_front_matter = True  # Skip lines before Abstract (title, subtitle, author)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Skip front matter (title/subtitle/author — already in preamble) ──
        if skip_front_matter:
            if stripped == '## Abstract':
                skip_front_matter = False
                # Fall through to section handler below
            else:
                i += 1
                continue

        # ── Skip references section (handled by BibTeX) ──
        if stripped == '## References':
            skip_until_section = True
            output.append('')
            output.append(r'\bibliographystyle{plainnat}')
            output.append(r'\bibliography{references}')
            output.append('')
            i += 1
            continue

        if skip_until_section:
            if stripped.startswith('## ') and stripped != '## References':
                skip_until_section = False
            else:
                i += 1
                continue

        # ── Code blocks ──
        if stripped.startswith('```') and not in_code_block:
            # Flush any pending list
            if in_list:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []
            # Flush any pending table
            if in_table:
                output.append(convert_table(table_lines))
                in_table = False
                table_lines = []

            in_code_block = True
            code_lang = stripped[3:].strip()
            code_lines = []

            # Handle mermaid diagrams specially
            if code_lang == 'mermaid':
                # Skip mermaid block, output placeholder
                while i < len(lines) and not (lines[i].strip() == '```' and i > 0 and lines[i-1].strip() != '```' + code_lang):
                    i += 1
                    if i < len(lines) and lines[i].strip() == '```':
                        break
                in_code_block = False
                output.append('')
                output.append(r'% [Mermaid diagram — render separately and include as figure]')
                output.append(r'% \begin{figure}[htbp]')
                output.append(r'%   \centering')
                output.append(r'%   \includegraphics[width=\columnwidth]{figures/diagram_placeholder.pdf}')
                output.append(r'%   \caption{Diagram.}')
                output.append(r'% \end{figure}')
                output.append('')
                i += 1
                continue

            i += 1
            continue

        if in_code_block:
            if stripped == '```':
                in_code_block = False
                # Output code block
                lang_map = {'python': 'Python', 'json': 'json', 'bash': 'bash', '': ''}
                lst_lang = lang_map.get(code_lang, '')
                output.append('')
                if lst_lang:
                    output.append(f'\\begin{{lstlisting}}[language={lst_lang}]')
                else:
                    output.append(r'\begin{lstlisting}')
                output.extend(code_lines)
                output.append(r'\end{lstlisting}')
                output.append('')
                code_lines = []
            else:
                code_lines.append(sanitize_code_line(line))
            i += 1
            continue

        # ── Tables ──
        if stripped.startswith('|') and '|' in stripped[1:]:
            if not in_table:
                # Flush any pending list
                if in_list:
                    output.extend(flush_list(list_items, list_type))
                    in_list = False
                    list_items = []
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            output.append(convert_table(table_lines))
            in_table = False
            table_lines = []
            # Don't increment i — reprocess current line

        # ── Horizontal rules ──
        if stripped == '---' or stripped == '***' or stripped == '___':
            if in_list:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []
            # Skip decorative rules (common after title/author in markdown)
            # Only emit a rule if surrounded by content
            i += 1
            continue

        # ── Section headers ──
        if stripped.startswith('#'):
            if in_list:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []

            level = len(stripped) - len(stripped.lstrip('#'))
            title = stripped.lstrip('#').strip()
            title = convert_inline(title)
            title = replace_citations(title, cite_map)

            # Strip leading section numbers ("1. Introduction" → "Introduction")
            title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', title)

            if level == 1:
                # Skip — title is already in preamble
                pass
            elif level == 2:
                if title == 'Abstract':
                    # Use abstract environment; collect lines until next section
                    output.append(r'\begin{abstract}')
                    i += 1
                    while i < len(lines):
                        next_stripped = lines[i].strip()
                        if next_stripped.startswith('## ') or next_stripped == '---':
                            break
                        if next_stripped:
                            converted = convert_inline(next_stripped)
                            converted = replace_citations(converted, cite_map)
                            output.append(converted)
                        else:
                            output.append('')
                        i += 1
                    output.append(r'\end{abstract}')
                    continue  # Don't increment i again
                elif title.startswith('Appendix'):
                    if 'Appendix A' in title or 'Appendix' == title.split(':')[0].strip():
                        output.append(r'\appendix')
                    output.append(f'\\section{{{title}}}')
                else:
                    output.append(f'\\section{{{title}}}')
            elif level == 3:
                output.append(f'\\subsection{{{title}}}')
            elif level == 4:
                output.append(f'\\subsubsection{{{title}}}')
            elif level == 5:
                output.append(f'\\paragraph{{{title}}}')
            i += 1
            continue

        # ── Display math ($$...$$) — pass through verbatim ──
        if stripped.startswith('$$'):
            if in_list:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []
            # Single-line display math: $$...$$
            if stripped.endswith('$$') and len(stripped) > 4:
                output.append(stripped)
                i += 1
                continue
            # Multi-line display math: collect until closing $$
            math_lines = [stripped]
            i += 1
            while i < len(lines):
                math_lines.append(lines[i].rstrip())
                if lines[i].strip().endswith('$$'):
                    break
                i += 1
            output.extend(math_lines)
            i += 1
            continue

        # ── Images ──
        if stripped.startswith('!['):
            if in_list:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []
            img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
            if img_match:
                caption = convert_inline(img_match.group(1))
                path = img_match.group(2)
                output.append('')
                output.append(r'\begin{figure}[htbp]')
                output.append(r'  \centering')
                output.append(f'  \\includegraphics[width=\\textwidth]{{{path}}}')
                output.append(f'  \\caption{{{caption}}}')
                output.append(r'\end{figure}')
                output.append('')
            i += 1
            continue

        # ── Lists ──
        list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)', line)
        if list_match:
            indent = len(list_match.group(1))
            marker = list_match.group(2)
            content = list_match.group(3)

            new_type = 'enumerate' if re.match(r'\d+\.', marker) else 'itemize'

            if not in_list:
                in_list = True
                list_type = new_type
                list_items = []

            list_items.append(content)
            i += 1
            continue
        elif in_list and stripped == '':
            # Empty line might end a list
            # Check if next non-empty line is also a list item
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and re.match(r'^(\s*)([-*+]|\d+\.)\s+', lines[j]):
                i += 1
                continue
            else:
                output.extend(flush_list(list_items, list_type))
                in_list = False
                list_items = []

        # ── Empty lines ──
        if stripped == '':
            output.append('')
            i += 1
            continue

        # ── Regular paragraph text ──
        if in_list:
            output.extend(flush_list(list_items, list_type))
            in_list = False
            list_items = []

        converted = convert_inline(stripped)
        converted = replace_citations(converted, cite_map)
        output.append(converted)
        i += 1

    # Flush any remaining state
    if in_list:
        output.extend(flush_list(list_items, list_type))
    if in_table:
        output.append(convert_table(table_lines))

    # Close document
    output.append('')
    output.append(r'\end{document}')

    # Post-process
    output = postprocess_latex(output)

    # Write output
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"  Generated {tex_path} ({len(output)} lines)")


def flush_list(items: List[str], list_type: str) -> List[str]:
    """Convert accumulated list items to LaTeX."""
    if not items:
        return []
    env = list_type or 'itemize'
    lines = [f'\\begin{{{env}}}']
    for item in items:
        converted = convert_inline(item)
        lines.append(f'  \\item {converted}')
    lines.append(f'\\end{{{env}}}')
    return lines


def generate_preamble() -> str:
    """Generate LaTeX preamble compatible with JAIR and TMLR."""
    return r"""%% ═══════════════════════════════════════════════════════════════════════════
%% PRECEPT: Planning Resilience via Experience, Context Engineering
%%          & Probing Trajectories
%%
%% LaTeX version — formatted for JAIR / TMLR submission
%% ═══════════════════════════════════════════════════════════════════════════
%%
%% VENUE SELECTION:
%% ────────────────
%% For JAIR:  Download jair.sty from https://www.jair.org/index.php/jair/about/submissions
%%            Replace \documentclass below with: \documentclass{article}
%%            Add: \usepackage{jair}
%%
%% For TMLR:  Download tmlr.sty from https://github.com/JmlrOrg/tmlr-style-file
%%            Replace \documentclass below with: \documentclass{article}
%%            Add: \usepackage{tmlr}
%%
%% Current:   Generic article class (compiles standalone)
%% ═══════════════════════════════════════════════════════════════════════════

\documentclass[11pt,a4paper]{article}

% ── Core packages ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[english]{babel}

% ── Page layout ──
\usepackage[margin=1in]{geometry}

% ── Math ──
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}

% ── Tables ──
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{tabularx}

% ── Figures ──
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{subcaption}
\usepackage{float}

% ── Code listings ──
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=none,
  backgroundcolor=\color{gray!5},
  keywordstyle=\color{blue!70!black},
  commentstyle=\color{green!50!black},
  stringstyle=\color{red!60!black},
  showstringspaces=false,
  tabsize=2,
  xleftmargin=0.5em,
  xrightmargin=0.5em,
}

% ── Colors ──
\usepackage[dvipsnames]{xcolor}

% ── Hyperlinks ──
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black]{hyperref}

% ── Citations (natbib for \citep/\citet) ──
\usepackage[round]{natbib}

% ── Misc ──
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{xspace}

% ── Theorem environments ──
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}{Remark}[section]

% ── Custom commands ──
\newcommand{\precept}{\textsc{Precept}\xspace}
\newcommand{\expel}{\textsc{ExpeL}\xspace}
\newcommand{\Pfirst}{P_1}
\newcommand{\Ptotal}{P_t}

% ═══════════════════════════════════════════════════════════════════════════

\title{PRECEPT: Planning Resilience via Experience, Context Engineering \& Probing Trajectories\\[0.5em]
\large A Unified Framework for Test-Time Adaptation with Compositional Rule Learning and Pareto-Guided Prompt Evolution}

\author{Arash Shahmansoori}

\date{}

\begin{document}

\maketitle
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent

    md_path = project_root / 'PRECEPT_PAPER.md'
    tex_path = project_root / 'PRECEPT_PAPER.tex'
    bib_path = project_root / 'references.bib'

    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    print("=" * 60)
    print("CONVERTING PRECEPT_PAPER.md → LaTeX")
    print("=" * 60)

    convert_md_to_latex(str(md_path), str(tex_path), str(bib_path))

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  • {tex_path}")
    print(f"  • {bib_path}")
    print(f"\nTo compile:")
    print(f"  pdflatex PRECEPT_PAPER.tex")
    print(f"  bibtex PRECEPT_PAPER")
    print(f"  pdflatex PRECEPT_PAPER.tex")
    print(f"  pdflatex PRECEPT_PAPER.tex")
    print(f"\nFor JAIR/TMLR: see comments at top of .tex file for template setup.")
