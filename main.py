import pandas as pd
import src.windpower_forecast
from pathlib import Path

# Make sure running under same folder each time by calling the path where main.py it is
main_dir = Path(__file__).resolve().parent

# Data path
input_folder_path = main_dir / "inputs"
traning_data_path = main_dir / "inputs" / "Location1.csv"

#df = src.windpower_forecast.forecaster_io.read_csv(traning_data_path)
#print(df)

src.windpower_forecast.GUI.launch_gui(input_folder_path)