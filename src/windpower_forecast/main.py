# src/windpower_forecast/main.py

from pathlib import Path

def _resolve_default_paths():
    base = Path(__file__).resolve().parents[1]
    inputs = base / "inputs"
    outputs = base / "outputs"
    outputs.mkdir(exist_ok=True)
    return inputs, outputs

def run():
    # Check tkinter
    try:
        import tkinter  # noqa
    except ImportError:
        raise ImportError(
            "tkinter is not installed.\n"
            "On Ubuntu run:\n"
            "  sudo apt-get install python3-tk"
        )

    from .GUI import ForecastApp

    inputs_dir, outputs_dir = _resolve_default_paths()

    app = ForecastApp(inputs_dir=inputs_dir, outputs_dir=outputs_dir)
    app.run()
