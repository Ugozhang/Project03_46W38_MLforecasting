# src/windpower_forecast/main.py

def run(inputs_dir=None, outputs_dir=None):
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

    app = ForecastApp(inputs_dir=inputs_dir, outputs_dir=outputs_dir)
    app.run()
