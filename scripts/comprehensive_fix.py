#!/usr/bin/env python3
"""
Comprehensive LaTeX quality fix for PRECEPT_PAPER.tex.
Addresses:
1. Switch all Mermaid figures from PDF to PNG (fixes blank fig1)
2. Per-figure sizing based on actual image dimensions
3. Remove escaped underscores inside lstlisting blocks
4. Float placement improvements
5. Ensure all tables fit within margins
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEX_FILE = PROJECT_ROOT / "PRECEPT_PAPER.tex"

# Actual PNG dimensions from re-render (width x height in pixels)
PNG_DIMS = {
    "fig1_mcp_architecture":      (1346, 3090),   # Very tall
    "fig1a_execution_flow":       (2368, 4732),   # Very tall
    "fig1c_knowledge_layer":      (1956, 700),    # Wide
    "fig2_simplified_pipeline":   (1968, 366),    # Wide, short
    "fig3_tier_hierarchy":        (1056, 188),    # Wide, very short
    "fig4_thompson_sampling":     (1408, 444),    # Wide
    "fig4a_evo_memory_lifecycle": (1480, 524),    # Wide
    "fig6_compass_architecture":  (1568, 152),    # Very wide, thin strip
    "fig7_pareto_selection":      (1568, 164),    # Very wide, thin strip
    "fig8_verified_prompt_evolution": (1768, 150),# Very wide, thin strip
    "fig9_dual_frequency_loop":   (1554, 986),    # Moderate
    "fig_exp2_compositional":     (1400, 1000),   # Near-square
    "fig_exp3_drift_recovery":    (1400, 1000),   # Near-square
    "fig_exp5_ablation":          (524, 1868),    # Very tall, narrow
    "fig10_positioning_summary":  (1000, 1000),   # Square
    "fig_d1_learning_curves":     (1400, 1000),   # Near-square
}

# LaTeX sizing per figure based on aspect ratio
def get_latex_size(name):
    """Return optimal LaTeX includegraphics options for each figure."""
    w, h = PNG_DIMS.get(name, (800, 600))
    ratio = h / w
    
    if name == "fig1_mcp_architecture":
        # Very tall architecture diagram - constrain height to fit page
        return r"width=0.75\textwidth,height=0.85\textheight,keepaspectratio"
    elif name == "fig1a_execution_flow":
        # Extremely tall - needs strong height constraint
        return r"width=0.85\textwidth,height=0.88\textheight,keepaspectratio"
    elif name == "fig_exp5_ablation":
        # Tall narrow ablation chart
        return r"width=0.45\textwidth,height=0.7\textheight,keepaspectratio"
    elif ratio < 0.15:
        # Very thin horizontal strips (pipeline diagrams)
        # These are intentionally short - use full width, natural height
        return r"width=\textwidth"
    elif ratio < 0.4:
        # Moderately short horizontal diagrams
        return r"width=0.92\textwidth"
    elif ratio > 1.5:
        # Tall diagrams
        return r"width=0.7\textwidth,height=0.75\textheight,keepaspectratio"
    elif 0.6 < ratio < 0.85:
        # Near-square charts (experimental results)
        return r"width=0.7\textwidth"
    else:
        # Default moderate
        return r"width=0.85\textwidth"


def fix_figure_includes(tex: str) -> str:
    """Switch all mermaid figure includes from PDF to PNG with proper sizing."""
    
    def replace_includegraphics(m):
        full_match = m.group(0)
        options = m.group(1)
        path = m.group(2)
        
        # Only fix mermaid figures
        if "figures/mermaid/" not in path:
            return full_match
        
        # Extract figure name
        name = path.replace("figures/mermaid/", "").replace(".pdf", "").replace(".png", "")
        
        # Get optimal sizing
        new_options = get_latex_size(name)
        
        # Switch to PNG
        new_path = path.replace(".pdf", ".png")
        
        return f"\\includegraphics[{new_options}]{{{new_path}}}"
    
    tex = re.sub(
        r'\\includegraphics\[([^\]]*)\]\{([^}]*)\}',
        replace_includegraphics,
        tex
    )
    return tex


def fix_lstlisting_underscores(tex: str) -> str:
    """Remove escaped underscores inside lstlisting environments.
    
    In lstlisting, _ should be plain _ (not \_).
    Also fix other escaping that shouldn't be in lstlisting.
    """
    
    def fix_listing_block(m):
        begin = m.group(1)  # \begin{lstlisting} possibly with options
        content = m.group(2)
        end = m.group(3)    # \end{lstlisting}
        
        # Unescape underscores: \_ -> _
        content = content.replace('\\_', '_')
        
        # Unescape ampersands: \& -> &
        content = content.replace('\\&', '&')
        
        # Unescape percent: \% -> %
        content = content.replace('\\%', '%')
        
        # Fix escaped braces that shouldn't be in lstlisting
        # But preserve \{ and \} that are part of pseudocode notation
        # Actually, in lstlisting, curly braces are literal
        # Keep { and } as-is
        
        # Fix double-escaped items
        content = content.replace('\\\\', '\\')
        
        # Remove any \textbf, \textit wrappers that leaked into code
        content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\textit\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\texttt\{([^}]*)\}', r'\1', content)
        
        # Fix common LaTeX commands that shouldn't be in code
        content = content.replace('\\leftarrow', '<-')
        content = content.replace('\\rightarrow', '->')
        content = content.replace('$\\times$', 'x')
        content = content.replace('\\times', 'x')
        content = content.replace('$\\geq$', '>=')
        content = content.replace('\\geq', '>=')
        content = content.replace('$\\leq$', '<=')
        content = content.replace('\\leq', '<=')
        content = content.replace('$\\Sigma$', 'Sum')
        content = content.replace('\\Sigma', 'Sum')
        content = content.replace('$\\kappa$', 'kappa')
        content = content.replace('\\kappa', 'kappa')
        content = content.replace('$\\theta$', 'theta')
        content = content.replace('\\theta', 'theta')
        content = content.replace('$\\alpha$', 'alpha')
        content = content.replace('\\alpha', 'alpha')
        content = content.replace('$\\beta$', 'beta')
        content = content.replace('\\beta', 'beta')
        content = content.replace('$\\lambda$', 'lambda')
        content = content.replace('\\lambda', 'lambda')
        content = content.replace('$\\delta$', 'delta')
        content = content.replace('\\delta', 'delta')
        content = content.replace('\\S{}', 'S')
        content = content.replace('\\S', 'S')
        
        # Remove stray $ signs around simple text in code
        content = re.sub(r'\$([a-zA-Z_]+)\$', r'\1', content)
        
        return f"{begin}{content}{end}"
    
    # Match lstlisting blocks (including optional arguments)
    tex = re.sub(
        r'(\\begin\{lstlisting\}(?:\[[^\]]*\])?)(.*?)(\\end\{lstlisting\})',
        fix_listing_block,
        tex,
        flags=re.DOTALL
    )
    
    return tex


def fix_float_placement(tex: str) -> str:
    """Improve float placement to reduce whitespace."""
    
    # Add float barrier support and placement tweaks to preamble
    # Check if already has the float spacing settings
    if '\\setlength{\\floatsep}' not in tex:
        # Add before \title
        float_settings = r"""
% ── Float spacing (reduce whitespace around figures/tables) ──
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{2pt}

% Allow more floats per page
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.7}
"""
        tex = tex.replace(
            r'\title{PRECEPT',
            float_settings + r'\title{PRECEPT'
        )
    
    return tex


def fix_table_overflow(tex: str) -> str:
    """Ensure all tables fit within text width."""
    
    # Check for tabular not wrapped in resizebox
    # Find \begin{tabular} not preceded by \resizebox
    lines = tex.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a tabular that's not already in resizebox
        if '\\begin{tabular}' in line:
            # Look back to see if previous non-empty line has \resizebox
            prev_idx = i - 1
            while prev_idx >= 0 and lines[prev_idx].strip() == '':
                prev_idx -= 1
            
            if prev_idx >= 0 and '\\resizebox' in lines[prev_idx]:
                # Already wrapped
                fixed_lines.append(line)
            else:
                # Need to wrap
                # Find matching \end{tabular}
                end_idx = i + 1
                depth = 1
                while end_idx < len(lines) and depth > 0:
                    if '\\begin{tabular}' in lines[end_idx]:
                        depth += 1
                    if '\\end{tabular}' in lines[end_idx]:
                        depth -= 1
                    end_idx += 1
                
                # Wrap the tabular
                fixed_lines.append('\\resizebox{\\textwidth}{!}{%')
                for j in range(i, end_idx):
                    fixed_lines.append(lines[j])
                fixed_lines.append('}%')
                i = end_idx
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def fix_long_lstlisting_lines(tex: str) -> str:
    """Truncate extremely long lines in lstlisting to prevent overflow."""
    MAX_LINE_LEN = 95  # characters
    
    def truncate_listing(m):
        begin = m.group(1)
        content = m.group(2)
        end = m.group(3)
        
        new_lines = []
        for line in content.split('\n'):
            if len(line) > MAX_LINE_LEN and not line.strip().startswith('//') and not line.strip().startswith('%'):
                # Truncate with continuation indicator
                # Try to break at a natural point
                cut = line[:MAX_LINE_LEN]
                # Try to break at last space
                space_idx = cut.rfind(' ')
                if space_idx > MAX_LINE_LEN * 0.6:
                    cut = cut[:space_idx]
                # Check if remaining is substantial
                remaining = line[len(cut):].strip()
                if remaining:
                    new_lines.append(cut)
                    new_lines.append('    ' + remaining)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        return begin + '\n'.join(new_lines) + end
    
    tex = re.sub(
        r'(\\begin\{lstlisting\}(?:\[[^\]]*\])?)(.*?)(\\end\{lstlisting\})',
        truncate_listing,
        tex,
        flags=re.DOTALL
    )
    return tex


def fix_separator_lines(tex: str) -> str:
    """Shorten overly long separator lines in lstlisting."""
    # Replace very long separator lines
    tex = re.sub(r'-{40,}', '-' * 70, tex)
    tex = re.sub(r'={40,}', '=' * 70, tex)
    return tex


def fix_blockquote(tex: str) -> str:
    """Convert markdown-style blockquotes to LaTeX."""
    # Fix any remaining > blockquotes
    lines = tex.split('\n')
    in_quote = False
    new_lines = []
    
    for line in lines:
        if line.startswith('> '):
            if not in_quote:
                new_lines.append('\\begin{quote}')
                in_quote = True
            new_lines.append(line[2:])
        else:
            if in_quote:
                new_lines.append('\\end{quote}')
                in_quote = False
            new_lines.append(line)
    
    if in_quote:
        new_lines.append('\\end{quote}')
    
    return '\n'.join(new_lines)


def remove_double_blank_lines(tex: str) -> str:
    """Remove excessive blank lines (more than 2 in a row)."""
    tex = re.sub(r'\n{4,}', '\n\n\n', tex)
    return tex


def main():
    print("Reading PRECEPT_PAPER.tex...")
    tex = TEX_FILE.read_text(encoding='utf-8')
    original_len = len(tex)
    
    print("\n[1/7] Fixing figure includes (PDF -> PNG + sizing)...")
    tex = fix_figure_includes(tex)
    
    print("[2/7] Fixing escaped underscores in lstlisting blocks...")
    tex = fix_lstlisting_underscores(tex)
    
    print("[3/7] Fixing float placement settings...")
    tex = fix_float_placement(tex)
    
    print("[4/7] Fixing separator lines...")
    tex = fix_separator_lines(tex)
    
    print("[5/7] Fixing blockquotes...")
    tex = fix_blockquote(tex)
    
    print("[6/7] Truncating long lstlisting lines...")
    tex = fix_long_lstlisting_lines(tex)
    
    print("[7/7] Cleaning up whitespace...")
    tex = remove_double_blank_lines(tex)
    
    # Write back
    TEX_FILE.write_text(tex, encoding='utf-8')
    new_len = len(tex)
    
    print(f"\nDone! {original_len} -> {new_len} characters")
    print(f"Delta: {new_len - original_len:+d} characters")
    
    # Verify all figures point to PNG
    pdf_refs = len(re.findall(r'figures/mermaid/.*\.pdf', tex))
    png_refs = len(re.findall(r'figures/mermaid/.*\.png', tex))
    print(f"\nFigure references: {png_refs} PNG, {pdf_refs} PDF (should be 0 PDF)")
    
    # Count remaining \_ in lstlisting blocks
    listing_underscores = 0
    for m in re.finditer(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', tex, re.DOTALL):
        listing_underscores += m.group(0).count('\\_')
    print(f"Escaped underscores in lstlisting: {listing_underscores} (should be 0)")


if __name__ == '__main__':
    main()
