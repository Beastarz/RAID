# Verification Environment

The explainability work is currently verified with the repository-local
virtual environment at `.venv/`, installed from the unpinned `requirements.txt`.

Recorded on 2026-09-01 on Windows:

```text
Python:          3.14.7 (64-bit)
Executable:      C:\Users\keithPC\Documents\RAID\.venv\Scripts\python.exe
torch:           2.13.0+cpu
torchvision:     0.28.0+cpu
albumentations:  2.0.8
numpy:           2.5.2
scikit-learn:    1.9.0
matplotlib:      3.11.1
gradio:          6.26.0
datasets:        5.0.1
PyYAML:          6.0.3
pytest:          9.1.1
Pillow:          12.3.0
```

Run project tests with:

```text
.\.venv\Scripts\python.exe -m pytest -q tests
```

The listed versions are the versions currently resolved in this Windows
environment. The requirements file remains intentionally unpinned; recreate
the environment and update this record when dependency versions materially
change. The earlier Wave 2.0 baseline was recorded under a macOS `.venv/bin`
path; that path and its Python 3.13.9 interpreter are historical, not the
current verification command.
