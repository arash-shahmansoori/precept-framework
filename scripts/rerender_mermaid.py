#!/usr/bin/env python3
"""
Re-render all Mermaid diagrams with proper per-diagram sizing.
Then convert all to high-quality PNG for LaTeX inclusion.
"""

import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MERMAID_DIR = PROJECT_ROOT / "figures" / "mermaid"
MMDC = PROJECT_ROOT / "node_modules" / ".bin" / "mmdc"

# Per-diagram render settings: (width, height) in pixels
# Height=0 means auto
RENDER_SETTINGS = {
    "fig1_mcp_architecture":      (1200, 0),    # Tall architecture - moderate width
    "fig1a_execution_flow":       (1200, 0),    # Very tall flowchart
    "fig1c_knowledge_layer":      (1200, 0),    # Wide knowledge layer
    "fig2_simplified_pipeline":   (1000, 0),    # Horizontal pipeline with branch
    "fig3_tier_hierarchy":        (600, 0),     # Simple 3-box hierarchy
    "fig4_thompson_sampling":     (1000, 0),    # Horizontal flow
    "fig4a_evo_memory_lifecycle": (1000, 0),    # Horizontal flow
    "fig6_compass_architecture":  (800, 0),     # Horizontal pipeline
    "fig7_pareto_selection":      (800, 0),     # Horizontal pipeline
    "fig8_verified_prompt_evolution": (900, 0), # Two-row flow
    "fig9_dual_frequency_loop":   (1000, 0),    # Moderate complexity
    "fig_exp2_compositional":     (800, 600),   # XY chart
    "fig_exp3_drift_recovery":    (800, 600),   # XY chart
    "fig_exp5_ablation":          (600, 800),   # Vertical bar chart
    "fig10_positioning_summary":  (700, 700),   # Quadrant chart
    "fig_d1_learning_curves":     (800, 600),   # XY chart
}


def render(name, width, height):
    """Render a single diagram to PNG."""
    mmd = MERMAID_DIR / f"{name}.mmd"
    png = MERMAID_DIR / f"{name}.png"

    if not mmd.exists():
        print(f"  [SKIP] {name}.mmd not found")
        return False

    config_path = MERMAID_DIR / "puppeteer-config.json"
    config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
    with open(config_path, 'w') as f:
        json.dump(config, f)

    cmd = [
        str(MMDC),
        "-i", str(mmd),
        "-o", str(png),
        "-w", str(width),
        "-b", "white",
        "-s", "2",  # Scale factor 2x for high DPI
        "--puppeteerConfigFile", str(config_path),
    ]

    if height > 0:
        cmd.extend(["-H", str(height)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Get resulting dimensions
            try:
                from PIL import Image
                im = Image.open(str(png))
                w, h = im.size
                print(f"  [OK]  {name}.png  ({w}x{h})")
            except ImportError:
                print(f"  [OK]  {name}.png")
            return True
        else:
            print(f"  [ERR] {name}: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name}")
        return False


def main():
    print("=" * 60)
    print("RE-RENDERING ALL MERMAID DIAGRAMS")
    print("=" * 60)

    success = 0
    total = len(RENDER_SETTINGS)

    for name, (w, h) in RENDER_SETTINGS.items():
        if render(name, w, h):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Rendered {success}/{total} diagrams")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
