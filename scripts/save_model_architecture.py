#!/usr/bin/env python3
"""
Save a high-level (depth=1) model overview of AntimicrobialPredictor using torchview
to plots/model_overview.png.

Requires:
  - System Graphviz (macOS: brew install graphviz)
  - Python packages: torchview, graphviz (pip install torchview graphviz)

Usage:
  python scripts/save_model_architecture.py
"""

import os
import sys
from pathlib import Path
import torch
from torchview import draw_graph

# Ensure project root is on PYTHONPATH for absolute import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from antimicrobial_predictor import AntimicrobialPredictor


def main():
    os.makedirs("plots", exist_ok=True)

    model = AntimicrobialPredictor()
    model.eval()

    # Dummy inputs (DNA=50, peptide=100) and high-level graph
    dna = torch.randint(0, 5, (1, 50), dtype=torch.long)
    pep = torch.randint(0, 21, (1, 100), dtype=torch.long)

    graph = draw_graph(
        model,
        input_data=(dna, pep),
        graph_name="model_overview",
        depth=1,
        expand_nested=False,
    )

    os.makedirs("plots", exist_ok=True)
    graph.visual_graph.render("plots/model_overview", format="png", cleanup=True)
    print("Saved plots/model_overview.png")


if __name__ == "__main__":
    main()


