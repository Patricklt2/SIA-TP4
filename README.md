# SIA - TP4 — Quick run instructions

Install (recommended venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
```

Run exercises:
- Europe (Kohonen + Oja):
```bash
python3 tp4/main_europa.py
```

- Patterns (Hopfield):
```bash
python3 tp4/main_hopfield.py
```

Outputs
- Figures and plots are saved in `results/`.
- Console prints show progress and brief summaries.

Notes
- Change or remove the random seed in scripts to vary results.

