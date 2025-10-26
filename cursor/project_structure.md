schrodinger-nn/
├── README.md
├── PRD.md
├── RULES.md
├── PROJECT_STRUCTURE.md
├── config/
│ ├── solver.yaml
│ ├── dataset.yaml
│ └── train.yaml
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── init.py
│ ├── config_loader.py
│ ├── solver/
│ │ └── nlse_solver.py
│ ├── data/
│ │ └── generate_dataset.py
│ ├── model/
│ │ └── mlp_schrodinger.py
│ ├── loss/
│ │ └── physics_loss.py
│ ├── train/
│ │ └── engine.py
│ ├── evaluate/
│ │ └── evaluate_model.py
│ └── utils/
│ ├── plotting.py
│ ├── metrics.py
│ └── seeds.py
├── tests/
│ ├── test_dataset.py
│ ├── test_model.py
│ └── test_loss.py
├── outputs/
│ └── plots/
├── requirements.txt
└── mlruns/