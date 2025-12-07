# src/windpower_forecast/main.py

def _check_tkinter():
    try:
        import tkinter  # noqa: F401
    except ImportError as e:
        msg = (
            "tkinter is not installed.\n\n"
            "If you are using Ubuntu/Debian, please install it with:\n"
            "  sudo apt-get install python3-tk\n\n"
            "After that, try running the program again."
        )
        raise ImportError(msg) from e


def run():
    _check_tkinter()

    from .GUI import ForecastApp  # import after checking tkinter

    app = ForecastApp()
    app.run()
