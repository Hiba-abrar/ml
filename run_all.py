"""Re-run the whole project end to end (EDA -> tuning -> test -> saved model).

    python run_all.py

Executes notebooks/f1_final_project.ipynb in place (outputs, figures, reports/results/*
and models/final_pipeline.joblib are regenerated). Takes roughly 10-20 minutes on 2 CPU cores.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path = Path(__file__).parent / "notebooks" / "f1_final_project.ipynb"
nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
nbformat.write(nb, nb_path)
print("done ->", nb_path)
