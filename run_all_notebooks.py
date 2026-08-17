import subprocess
from pathlib import Path

NOTEBOOKS = [
    "01_data_cleaning.ipynb",
    "02_eda.ipynb",
    "03_eda_after_outlier_removal.ipynb",
    "04_ml_training.ipynb",
    "05_additional_outlier_experiments.ipynb",
    "06_additional_ml_training.ipynb",
]

for notebook in NOTEBOOKS:
    path = Path(notebook)

    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    print(f"\n{'=' * 60}")
    print(f"Running: {notebook}")
    print(f"{'=' * 60}")

    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(path),
        ],
        check=True,
    )

print("\nAll notebooks completed successfully.")