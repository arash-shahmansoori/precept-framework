#!/usr/bin/env python3
"""
Post-process PRECEPT_PAPER.tex to fix all remaining issues:
1. Fix citation formatting (multi-citations, stray parentheses)
2. Replace Mermaid placeholders with actual figure includes
3. Fix wide tables with resizebox
4. Fix lstlisting underscore display
5. Add adjustbox package
"""

import re
from pathlib import Path

TEX_PATH = Path(__file__).parent.parent / "PRECEPT_PAPER.tex"

# ─── Mermaid diagram → figure mapping ───
# Maps the Nth mermaid placeholder (1-indexed) to figure info
MERMAID_FIGURES = [
    {
        "file": "fig1_mcp_architecture",
        "caption": "PRECEPT MCP Architecture. The Client (\\texttt{precept\\_agent.py}) orchestrates task execution with COMPASS enhancements. Communication occurs via stdio/JSONRPC to the Server (\\texttt{precept\\_mcp\\_server.py}), which exposes 70+ MCP tools for memory retrieval, rule management, conflict resolution, and domain-specific actions.",
        "label": "fig:mcp_arch",
        "width": "0.95\\textwidth",
    },
    {
        "file": "fig1a_execution_flow",
        "caption": "Complete execution flow of the PRECEPT agent. The pipeline comprises seven phases: (1) Task Parsing, (2) COMPASS Evaluation, (3) Context Retrieval (three modes), (4) Solution Derivation, (5) Action Execution, (6) Outcome Processing with threshold-based invalidation ($\\theta=2$), and (7) Knowledge Update. Dashed arrows indicate retry loops.",
        "label": "fig:exec_flow",
        "width": "\\textwidth",
    },
    {
        "file": "fig1c_knowledge_layer",
        "caption": "Knowledge Layer with three retrieval modes: semantic similarity $O(\\log n)$, exact-match $O(1)$ via dictionary lookup, and hybrid BM25+semantic.",
        "label": "fig:knowledge_layer",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig2_simplified_pipeline",
        "caption": "Simplified PRECEPT pipeline. Extract condition key $\\kappa$ $\\rightarrow$ Retrieve (3 modes) $\\rightarrow$ Compose (tier sort) $\\rightarrow$ Execute $\\rightarrow$ Handle success/failure with threshold-based invalidation.",
        "label": "fig:simplified_pipeline",
        "width": "\\textwidth",
    },
    {
        "file": "fig3_tier_hierarchy",
        "caption": "Tier dominance hierarchy. Higher tiers always override lower tiers in compositional resolution.",
        "label": "fig:tier_hierarchy",
        "width": "0.6\\textwidth",
    },
    {
        "file": "fig4_thompson_sampling",
        "caption": "Type I conflict resolution pipeline. Ensemble voting triggers Bayesian resolution via Thompson Sampling.",
        "label": "fig:thompson_sampling",
        "width": "0.9\\textwidth",
    },
    {
        "file": "fig4a_evo_memory_lifecycle",
        "caption": "Type II (rule drift) handling. Soft confidence decay precedes hard invalidation at $\\theta=2$ failures.",
        "label": "fig:evo_memory",
        "width": "0.9\\textwidth",
    },
    {
        "file": "fig6_compass_architecture",
        "caption": "COMPASS pipeline: Complexity Analysis $\\rightarrow$ Smart Rollout Allocation $\\rightarrow$ Candidate Evaluation $\\rightarrow$ Pareto Selection.",
        "label": "fig:compass_arch",
        "width": "\\textwidth",
    },
    {
        "file": "fig7_pareto_selection",
        "caption": "Pareto selection: (1) filter dominated candidates, (2) select from front by hypervolume contribution.",
        "label": "fig:pareto_selection",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig8_verified_prompt_evolution",
        "caption": "Verified evolution signal flow. The agent predicts a solution, the environment verifies it internally, and only binary success/failure signals are used for evolution---the agent never sees expected solutions.",
        "label": "fig:verified_evolution",
        "width": "0.9\\textwidth",
    },
    {
        "file": "fig9_dual_frequency_loop",
        "caption": "COMPASS Dual-Frequency Control Loop. The high-frequency loop runs at every agent step for real-time safety monitoring. The low-frequency loop runs only on trigger events for strategic re-planning.",
        "label": "fig:dual_freq",
        "width": "0.9\\textwidth",
    },
    {
        "file": "fig_exp2_compositional",
        "caption": "Compositional generalization results. PRECEPT (dark) vs Full Reflexion (light). PRECEPT achieves 100\\% $P_1$ on 2-way logistics compositions.",
        "label": "fig:exp2_compositional",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig_exp3_drift_recovery",
        "caption": "Drift adaptation curves. PRECEPT recovers via \\texttt{record\\_rule\\_failure()} invalidation; baselines remain stuck at 0\\% for 3 encounters.",
        "label": "fig:exp3_drift",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig_exp5_ablation",
        "caption": "Component contribution ranking by $P_1$ impact. Compositional stacking and hybrid retrieval provide the largest gains.",
        "label": "fig:exp5_ablation",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig10_positioning_summary",
        "caption": "PRECEPT occupies a unique position: high sample efficiency ($\\beta=3$) AND compositional generalization ($O(2^N)$ coverage). Prior methods cluster in low-efficiency and/or non-compositional regions.",
        "label": "fig:positioning",
        "width": "0.85\\textwidth",
    },
    {
        "file": "fig_d1_learning_curves",
        "caption": "Learning efficiency comparison. PRECEPT reaches 90\\%+ $P_1$ with $\\sim$20 episodes. Baselines plateau at 50--55\\%.",
        "label": "fig:learning_curves",
        "width": "0.85\\textwidth",
    },
]


def fix_all_issues(tex_path: Path):
    with open(tex_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    result = []
    mermaid_idx = 0
    i = 0
    preamble_fixed = False

    while i < len(lines):
        line = lines[i]

        # ─── Fix 1: Add adjustbox and fix lstlisting in preamble ───
        if not preamble_fixed and line.strip() == r'\usepackage{enumitem}':
            result.append(line)
            result.append(r'\usepackage{adjustbox}')
            preamble_fixed = True
            i += 1
            continue

        # ─── Fix 2: Add literate settings for lstlisting underscores ───
        if line.strip() == r'xleftmargin=0.5em,':
            result.append(line)
            # Check if next line is xrightmargin
            if i + 1 < len(lines) and 'xrightmargin' in lines[i + 1]:
                result.append(lines[i + 1])
                # Add literate settings after xrightmargin
                result.append(r'  literate={\_}{\_}1{->}{->}1{<-}{<-}1,')
                i += 2
                continue

        # ─── Fix 3: Replace Mermaid placeholders with actual figures ───
        if line.strip() == r'% [Mermaid diagram — render separately and include as figure]':
            # Skip the entire placeholder block (next 6 commented lines)
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('%') or lines[j].strip() == ''):
                j += 1

            # Insert actual figure
            if mermaid_idx < len(MERMAID_FIGURES):
                fig = MERMAID_FIGURES[mermaid_idx]
                result.append('')
                result.append(r'\begin{figure}[htbp]')
                result.append(r'  \centering')
                result.append(f'  \\includegraphics[width={fig["width"]}]{{figures/mermaid/{fig["file"]}.pdf}}')
                result.append(f'  \\caption{{{fig["caption"]}}}')
                result.append(f'  \\label{{{fig["label"]}}}')
                result.append(r'\end{figure}')
                result.append('')
                mermaid_idx += 1
            else:
                result.append(line)  # Keep placeholder if we run out

            i = j
            continue

        # ─── Fix 4: Fix abstract multi-citation ───
        # "(Shinn et al., 2023) Zhao et al. (2023))" → "\citep{shinn2023,zhao2023}"
        if r'\citep{shinn2023}' in line and r'\citet{zhao2023})' in line:
            line = line.replace(
                r'\citep{shinn2023} \citet{zhao2023})',
                r'\citep{shinn2023,zhao2023}'
            )

        # ─── Fix 5: Wrap wide tables with resizebox ───
        if line.strip() == r'\begin{tabular}' or (r'\begin{tabular}{' in line.strip()):
            # Count columns
            col_match = re.search(r'\\begin\{tabular\}\{([^}]+)\}', line)
            if col_match:
                col_spec = col_match.group(1)
                n_cols = col_spec.count('l') + col_spec.count('c') + col_spec.count('r')
                # Wrap tables with 5+ columns in resizebox
                if n_cols >= 5:
                    # Find the \centering or \small before this
                    # Insert resizebox wrapper
                    result.append(r'\resizebox{\textwidth}{!}{%')
                    result.append(line)
                    # Find matching \end{tabular}
                    i += 1
                    while i < len(lines):
                        result.append(lines[i])
                        if r'\end{tabular}' in lines[i]:
                            result.append('}')  # Close resizebox
                            i += 1
                            break
                        i += 1
                    continue

        # ─── Fix 6: Remove duplicate figure captions ───
        # After a figure, the markdown had "**Figure X.**" as a separate paragraph
        # which got converted to \textbf{Figure X.} — skip these if they follow a \end{figure}
        if (line.strip().startswith(r'\textbf{Figure') and
            len(result) > 0 and
            any(r'\end{figure}' in result[k] for k in range(max(0, len(result)-3), len(result)))):
            i += 1
            continue

        result.append(line)
        i += 1

    # Write result
    with open(tex_path, 'w') as f:
        f.write('\n'.join(result))

    print(f"Fixed {tex_path}")
    print(f"  - Replaced {mermaid_idx} Mermaid placeholders with figures")
    print(f"  - Added adjustbox package")
    print(f"  - Added lstlisting literate settings")
    print(f"  - Fixed multi-citation in abstract")
    print(f"  - Wrapped wide tables with resizebox")


if __name__ == '__main__':
    fix_all_issues(TEX_PATH)
