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
python3 tp4/main_patrones.py
```

Outputs
- Figures and plots are saved in `results/` and `results/patterns/`.
- Console prints show progress and brief summaries.

Notes
- Change or remove the random seed in scripts to vary results.
- If a required package is missing, install it with `pip install <package>`.
