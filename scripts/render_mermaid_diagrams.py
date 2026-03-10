#!/usr/bin/env python3
"""
Extract all Mermaid diagrams from PRECEPT_PAPER.md and render them to PDF/PNG.
"""

import re
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MD_PATH = PROJECT_ROOT / "PRECEPT_PAPER.md"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "mermaid"
MMDC = PROJECT_ROOT / "node_modules" / ".bin" / "mmdc"


def extract_mermaid_blocks(md_path: Path):
    """Extract mermaid code blocks with their preceding caption context."""
    with open(md_path, 'r') as f:
        content = f.read()

    blocks = []
    # Find all ```mermaid ... ``` blocks
    pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
    lines = content.split('\n')

    block_idx = 0
    i = 0
    while i < len(lines):
        if lines[i].strip() == '```mermaid':
            # Collect the mermaid content
            mermaid_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '```':
                mermaid_lines.append(lines[i])
                i += 1

            mermaid_content = '\n'.join(mermaid_lines)

            # Look for figure caption after the closing ```
            caption = ""
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines):
                line = lines[j].strip()
                if line.startswith('**Figure') or line.startswith('Figure'):
                    caption = line.strip('*').strip()

            # Also look for context BEFORE the block
            context = ""
            k = i - len(mermaid_lines) - 2  # Line before ```mermaid
            while k >= 0 and lines[k].strip() == '':
                k -= 1
            if k >= 0:
                context = lines[k].strip()

            block_idx += 1
            blocks.append({
                'index': block_idx,
                'content': mermaid_content,
                'caption': caption,
                'context': context,
                'line': i - len(mermaid_lines),
            })
        i += 1

    return blocks


def determine_figure_name(block):
    """Generate a descriptive filename for the diagram."""
    idx = block['index']
    caption = block['caption'].lower()
    context = block['context'].lower()

    # Map based on context/caption
    name_map = {
        1: "fig1_mcp_architecture",
        2: "fig1a_execution_flow",
        3: "fig1c_knowledge_layer",
        4: "fig2_simplified_pipeline",
        5: "fig3_tier_hierarchy",
        6: "fig4_thompson_sampling",
        7: "fig4a_evo_memory_lifecycle",
        8: "fig6_compass_architecture",
        9: "fig7_pareto_selection",
        10: "fig8_verified_prompt_evolution",
        11: "fig9_dual_frequency_loop",
        12: "fig_exp2_compositional",
        13: "fig_exp3_drift_recovery",
        14: "fig_exp5_ablation",
        15: "fig10_positioning_summary",
        16: "fig_d1_learning_curves",
    }

    return name_map.get(idx, f"fig_mermaid_{idx}")


def render_diagram(block, output_dir: Path):
    """Render a single mermaid diagram to PDF and PNG."""
    name = determine_figure_name(block)
    mmd_file = output_dir / f"{name}.mmd"
    pdf_file = output_dir / f"{name}.pdf"
    png_file = output_dir / f"{name}.png"

    # Write mermaid source
    with open(mmd_file, 'w') as f:
        f.write(block['content'])

    # Render to PNG (PDF rendering with mmdc can be unreliable)
    try:
        result = subprocess.run(
            [str(MMDC), "-i", str(mmd_file), "-o", str(png_file),
             "-w", "2400", "-b", "white",
             "--puppeteerConfigFile", str(output_dir / "puppeteer-config.json")],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"  [OK]  {name}.png")
        else:
            print(f"  [ERR] {name}: {result.stderr[:200]}")

        # Also render to PDF
        result_pdf = subprocess.run(
            [str(MMDC), "-i", str(mmd_file), "-o", str(pdf_file),
             "-w", "2400", "-b", "white",
             "--puppeteerConfigFile", str(output_dir / "puppeteer-config.json")],
            capture_output=True, text=True, timeout=60
        )
        if result_pdf.returncode == 0:
            print(f"  [OK]  {name}.pdf")
        else:
            print(f"  [WARN] {name}.pdf failed, PNG available")

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

    return name


def main():
    print("=" * 60)
    print("EXTRACTING AND RENDERING MERMAID DIAGRAMS")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create puppeteer config (needed for mmdc)
    config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
    with open(OUTPUT_DIR / "puppeteer-config.json", 'w') as f:
        json.dump(config, f)

    # Extract blocks
    blocks = extract_mermaid_blocks(MD_PATH)
    print(f"\nFound {len(blocks)} Mermaid diagrams\n")

    # Render each
    figure_names = []
    for block in blocks:
        name = render_diagram(block, OUTPUT_DIR)
        figure_names.append(name)

    print(f"\n{'=' * 60}")
    print(f"RENDERED {len(figure_names)} DIAGRAMS to {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Print LaTeX include snippets
    print("\nLaTeX include snippets:")
    for i, (block, name) in enumerate(zip(blocks, figure_names)):
        caption = block['caption'] or f"Diagram {i+1}"
        print(f"""
\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{{figures/mermaid/{name}.png}}
  \\caption{{{caption}}}
\\end{{figure}}
""")


if __name__ == '__main__':
    main()
