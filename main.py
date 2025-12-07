# Project03_46W38_MLforecasting/main.py

from pathlib import Path
from src.windpower_forecast.main import run

def main():
    project_root = Path(__file__).resolve().parent
    inputs_dir = project_root / "inputs"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    run(inputs_dir=inputs_dir, outputs_dir=outputs_dir)

if __name__ == "__main__":
    main()
