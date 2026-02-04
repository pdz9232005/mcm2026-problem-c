# MCM 2026 Problem C – Voting System Analysis

This repository contains the code and outputs used in our MCM 2026 Problem C submission.

## Structure

- `first.py` : Task 1 Bayesian model (fan vote estimation)
- `task234.py` : Task 2–4 simulations and rule evaluation
- `2026_MCM_Problem_C_Data.csv` : Original dataset
- `outputs/` : Task 1 outputs (posterior summaries, figures)
- `outputs_task234/` : Task 2–4 outputs (comparisons, replays, grids)

## How to Run

1. Install dependencies:
```bash
pip install numpy pandas matplotlib pymc arviz
```

2. Run Task 1:
```bash
python first.py
```

3. Run Task 2–4:
```bash
python task234.py
```

All figures and CSV summaries will be generated into outputs/ and outputs_task234/.

## Note
Due to time and environment constraints, random seeds and runtime may cause minor numerical differences, but all qualitative conclusions remain stable.

## License
For academic and educational use only.
