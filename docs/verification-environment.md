# Verification Environment

The explainability work is verified with the repository-local virtual
environment at `.venv/`. The environment was created from the available Python
3.13 interpreter and installed with `requirements.txt`.

Recorded on 2026-08-31:

```text
Python:          3.13.9
Executable:      /Users/keith/Documents/RAID/.venv/bin/python
torch:           2.13.0
torchvision:     0.28.0
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
.venv/bin/python -m pytest -q
```

The listed versions are the versions resolved when this environment was
created. The requirements file remains intentionally unpinned; recreate the
environment and update this record when dependency versions materially change.
