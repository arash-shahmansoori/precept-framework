#!/usr/bin/env python3
"""
Fix math-mode issues in algorithm environments.
The auto-converter incorrectly wrapped code identifiers in $...$.
This script fixes:
1. $X_name \gets Y_value$ -> X\_name $\gets$ Y\_value  
2. Double-escaped underscores \\_ -> \_
3. Pipe chars | in \If conditions
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEX_FILE = PROJECT_ROOT / "PRECEPT_PAPER.tex"


def fix_state_assignment(m):
    """Fix \State $X \gets Y$ patterns."""
    prefix = m.group(1)  # \State or similar
    math_content = m.group(2)  # content inside $...$
    suffix = m.group(3) or ''  # anything after (like \Comment)
    
    # Check if this contains \gets (assignment)
    if '\\gets' in math_content:
        parts = math_content.split('\\gets', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        
        # Escape underscores in text parts
        def escape_text(s):
            # Don't escape inside existing LaTeX commands
            s = re.sub(r'(?<!\\)_', r'\\_', s)
            return s
        
        lhs = escape_text(lhs)
        rhs = escape_text(rhs)
        
        return f'{prefix}{lhs} $\\gets$ {rhs}{suffix}'
    
    # Not an assignment - just escape underscores and remove $ wrapping
    content = re.sub(r'(?<!\\)_', r'\\_', math_content)
    return f'{prefix}{content}{suffix}'


def fix_algorithm_content(tex):
    """Fix all algorithm blocks in the tex content."""
    
    # Find algorithm blocks
    alg_pattern = r'(\\begin\{algorithm\}.*?\\end\{algorithm\})'
    
    def fix_single_algorithm(m):
        block = m.group(0)
        
        # Fix double-escaped underscores: \\_ -> \_
        block = block.replace('\\\\_', '\\_')
        
        # Fix \State $...$ patterns  
        # Match: \State $content$  possibly followed by \Comment
        block = re.sub(
            r'(\\State\s+)\$([^$]+)\$(\s*\\Comment\{[^}]*\})?',
            fix_state_assignment,
            block
        )
        
        # Fix \State \Return $content$
        def fix_return(m):
            prefix = m.group(1)
            content = m.group(2)
            suffix = m.group(3) or ''
            content = re.sub(r'(?<!\\)_', r'\\_', content)
            return f'{prefix}{content}{suffix}'
        
        block = re.sub(
            r'(\\State\\s*\\Return\s+)\$([^$]+)\$(\s*\\Comment\{[^}]*\})?',
            fix_return,
            block
        )
        
        # Fix remaining $identifier_with_underscores$ patterns
        # These are identifiers that should be in text mode
        def fix_dollar_ident(m):
            content = m.group(1)
            # If it's actual math (has \alpha, \beta, \geq, \leq, \times, +, -, fractions)
            if any(cmd in content for cmd in ['\\alpha', '\\beta', '\\theta', '\\lambda',
                                               '\\geq', '\\leq', '\\times', '\\cdot',
                                               '\\frac', '\\sum', '\\prod', '\\sim',
                                               '\\arg\\max', '\\exp', '\\text{', '\\to',
                                               '> ', '< ', '= ']):
                return f'${content}$'
            # Simple identifier - move to text mode
            content = re.sub(r'(?<!\\)_', r'\\_', content)
            return content
        
        block = re.sub(r'\$([^$]{1,80})\$', fix_dollar_ident, block)
        
        # Fix pipe characters | in \If conditions (use \lvert \rvert instead)
        block = re.sub(r'\\If\{([^}]*)\|([^}]*)\|([^}]*)\}', 
                       lambda m: f'\\If{{{m.group(1)}$|${m.group(2)}$|${m.group(3)}}}',
                       block)
        
        # Fix any remaining double-escapes
        block = block.replace('\\\\_', '\\_')
        
        return block
    
    tex = re.sub(alg_pattern, fix_single_algorithm, tex, flags=re.DOTALL)
    return tex


def main():
    print("Reading PRECEPT_PAPER.tex...")
    tex = TEX_FILE.read_text(encoding='utf-8')
    
    tex = fix_algorithm_content(tex)
    
    TEX_FILE.write_text(tex, encoding='utf-8')
    
    # Count algorithm blocks
    alg_count = len(re.findall(r'\\begin\{algorithm\}', tex))
    print(f"Processed {alg_count} algorithm environments")
    
    # Check for remaining issues
    alg_blocks = re.findall(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', tex, re.DOTALL)
    issues = 0
    for block in alg_blocks:
        # Find $..$ blocks with bare underscores
        for m in re.finditer(r'\$([^$]+)\$', block):
            content = m.group(1)
            bare = re.findall(r'(?<!\\)_(?!\\)', content)
            if bare:
                # Check if it's actual math (subscript is ok)
                # Skip if near \alpha, \beta etc
                if not any(cmd in content for cmd in ['\\alpha', '\\beta', '\\theta', '\\gets']):
                    issues += len(bare)
                    print(f"  Potential issue: ${content}$")
    
    print(f"Potential remaining issues: {issues}")


if __name__ == '__main__':
    main()
