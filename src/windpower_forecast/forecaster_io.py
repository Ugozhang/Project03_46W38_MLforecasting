# Using GUI presenting the all project
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
from datetime import datetime, time
from pathlib import Path
import pandas as pd
from . import forecaster_plot as fc_plt
from .models import train_random_forest as rf_train
from . import data

def launch_gui(inputs_dir: Path, outputs_dir: Path | None = None):
    """
    Launch a GUI for selecting CSV files, choosing variables and date ranges,
    and plotting timeseries.

    Parameters
    ----------
    inputs_dir : Path
        Path to the inputs folder (provided by main.py).
    outputs_dir : Path, optional
        Path to outputs folder. GUI does not save plots unless extended.
    """

    # cache to store loaded CSVs
    DATA_CACHE: dict[str, pd.DataFrame] = {}

    # Data Files list (csv only in this project)
    def list_csv_files() -> list[str]:
        """List CSV files under the provided inputs directory."""
        return sorted([p.name for p in inputs_dir.glob("*.csv")])

    # Load File-df as dict
    def load_data(filename: str) -> pd.DataFrame:
        """Load CSV with caching."""
        if filename not in DATA_CACHE:
            df = pd.read_csv(inputs_dir / filename, parse_dates=["Time"])
            df = df.sort_values("Time").reset_index(drop=True)
            DATA_CACHE[filename] = df
        return DATA_CACHE[filename]

    # Create Variant List for plotting by filtering as numeric data column
    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="number").columns.tolist()

    # 這兩個用來記錄目前 CSV 的時間範圍（給日曆限制用）
    time_min = None
    time_max = None

    # ---------------- Helper: Calendar ---------------- #

    def open_calendar(target_var: tk.StringVar, initial_date: str | None = None):
        """開一個 Toplevel 月曆視窗，選完日期寫回 target_var"""

        nonlocal time_min, time_max

        top = tk.Toplevel(root)
        top.title("Select date")
        top.grab_set()  # 抓住焦點，避免點到後面

        # 決定初始日期
        if initial_date:
            try:
                init_dt = datetime.strptime(initial_date, "%Y-%m-%d").date()
            except ValueError:
                init_dt = None
        else:
            init_dt = None

        # mindate / maxdate is bounded in time range of CSV file (if read)
        mindate = time_min.date() if time_min is not None else None
        maxdate = time_max.date() if time_max is not None else None

        cal_kwargs = {
            "selectmode": "day",
            "date_pattern": "yyyy-mm-dd",
        }
        if mindate is not None:
            cal_kwargs["mindate"] = mindate
        if maxdate is not None:
            cal_kwargs["maxdate"] = maxdate
        if init_dt is not None:
            cal_kwargs["year"] = init_dt.year
            cal_kwargs["month"] = init_dt.month
            cal_kwargs["day"] = init_dt.day

        cal = Calendar(top, **cal_kwargs)
        cal.pack(padx=10, pady=10)

        def on_ok():
            picked = cal.get_date()  # 字串，已經是 yyyy-mm-dd
            target_var.set(picked)
            top.destroy()

        ttk.Button(top, text="OK", command=on_ok).pack(pady=5)

    # ---------------- GUI Logic Handlers ---------------- #

    def on_file_selected(event=None):
        """
        Run while a CSV file is selected.
        ------
        1. Read file
        2. Purse initial time and end time
        3. Set Default time value for GUI column
        """
        nonlocal time_min, time_max

        filename = combo_file.get()
        if not filename:
            return

        try:
            df = load_data(filename)

            tmin = df["Time"].min()
            tmax = df["Time"].max()
            time_min, time_max = tmin, tmax

            # Edit the Time range of file on root window
            label_range_var.set(f"Time range:\n{tmin} → {tmax}")

            # update variable dropdown
            num_cols = get_numeric_columns(df)
            combo_var["values"] = num_cols
            if num_cols:
                combo_var.set(num_cols[0])
            else:
                combo_var.set("")

            # Defaulte Date and Time : From CSV time range
            start_date_var.set(tmin.date().isoformat())
            end_date_var.set(tmax.date().isoformat())

            entry_start_hour.delete(0, tk.END)
            entry_start_min.delete(0, tk.END)
            entry_end_hour.delete(0, tk.END)
            entry_end_min.delete(0, tk.END)

            entry_start_hour.insert(0, "00")
            entry_start_min.insert(0, "00")
            entry_end_hour.insert(0, "23")
            entry_end_min.insert(0, "00")

        except Exception as e:
            messagebox.showerror("Error loading file", str(e))

    def run_plot():
        """
        Plot data.
        ------
        1. Purse Time(index), error catches.
        2. Call plot function from forecaster_plot
        """
        filename = combo_file.get()
        varname = combo_var.get()

        if not filename:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        if not varname:
            messagebox.showerror("Error", "Please select a variable.")
            return

        try:
            df = load_data(filename)

            # 解析日期字串
            if not start_date_var.get() or not end_date_var.get():
                raise ValueError("Please select both start and end dates.")

            start_date = datetime.strptime(start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_var.get(), "%Y-%m-%d").date()

            # 解析時間
            sh, sm = int(entry_start_hour.get()), int(entry_start_min.get())
            eh, em = int(entry_end_hour.get()), int(entry_end_min.get())

            start_dt = datetime.combine(start_date, time(sh, sm))
            end_dt = datetime.combine(end_date, time(eh, em))

            if start_dt > end_dt:
                raise ValueError("Start time cannot be after end time.")

            if start_dt < df["Time"].min() or end_dt > df["Time"].max():
                raise ValueError("Selected period is outside the CSV time range.")

            # filter data
            mask = (df["Time"] >= start_dt) & (df["Time"] <= end_dt)
            sub = df.loc[mask]

            if sub.empty:
                raise ValueError("No data in this time window.")

            # plot
            fc_plt.single_var_plot(sub["Time"], sub[varname], xlabel = "Time", ylabel = varname, title = f"{filename} – {varname}\n{start_dt} → {end_dt}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---- GUI ML training section ---- 
    def on_train_random_forest(split_ratio_var, status_var):
        # validate split ratio 
        try:
            split_ratio = float(split_ratio_var.get())
            if not (0 < split_ratio < 1):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "split ratio must between 0~1, e.g 0.8.")
            return
        
        # Using CSV chosen for ML
        filename = combo_file.get()
        if not filename:
            messagebox.showerror("Error", "Select CSV files above first.")
            return
        df_ML = data.transform_features(load_data(filename))

        # Show simple "progress" info (just text)
        train_pct = int(split_ratio * 100)
        test_pct = 100 - train_pct
        status_var.set(f"Training Random Forest...\n")
        root.update_idletasks()   # force GUI to refresh text

        # split index
        split_idx = int(len(df_ML) * split_ratio)
        train_df = df_ML.iloc[:split_idx].copy()
        test_df = df_ML.iloc[split_idx:].copy()

        # Assign feature columns for training
        feature_cols = [
            # Meteorology / scalar weather
            "temperature_2m",
            "relativehumidity_2m",
            "dewpoint_2m",
            #"windspeed_10m",
            #"windspeed_100m",
            "windgusts_10m",

            # Wind vectors (direction + speed combined)
            "u_10m",
            "v_10m",
            "u_100m",
            "v_100m",

            # Vertical wind profile info
            "delta_ws",      # = ws_100m - ws_10m

            # Time features (cyclic)
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
        ]
        target_col = "Power"

        model, scores = rf_train(train_df, test_df, feature_cols, target_col)

        # Update GUI label with metrics
        status_var.set(
            f"Random Forest done.\n"
            f"Train/Test split: {train_pct}% / {test_pct}%\n"
            f"MSE : {scores['mse']:.3f}\n"
            f"MAE : {scores['mae']:.3f}\n"
            f"RMSE: {scores['rmse']:.3f}"
        )


    # ---------------- Build GUI ---------------- #
    """
        GUI main
    """
    root = tk.Tk()
    root.title("Wind Power Timeseries Viewer")
    root.geometry("520x600")

    # ==== 1. File selection frame ====
    frame_file = ttk.LabelFrame(root, text="1. Choose CSV file")
    frame_file.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_file, text="CSV file:").grid(row=0, column=0, padx=5, pady=5)
    combo_file = ttk.Combobox(frame_file, width=40, state="readonly")
    combo_file["values"] = list_csv_files()
    combo_file.grid(row=0, column=1, padx=5, pady=5)
    combo_file.bind("<<ComboboxSelected>>", on_file_selected)

    # Time range of CSV file (Default statu)
    label_range_var = tk.StringVar(value="Time range: (select a file)")
    ttk.Label(frame_file, textvariable=label_range_var).grid(
        row=1, column=0, columnspan=2, padx=5, pady=5
    )
    
    # ==== 2. variable selection frame ====
    frame_var = ttk.LabelFrame(root, text="2. Choose variable")
    frame_var.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_var, text="Variable:").grid(row=0, column=0, padx=5, pady=5)
    combo_var = ttk.Combobox(frame_var, width=30, state="readonly")
    combo_var.grid(row=0, column=1, padx=5, pady=5)

    # ==== 3. date selection frame ====
    frame_time = ttk.LabelFrame(root, text="3. Choose time period")
    frame_time.pack(fill="x", padx=10, pady=5)

    # Carry DateString by StringVar
    start_date_var = tk.StringVar()
    end_date_var = tk.StringVar()

    # start date/time
    ttk.Label(frame_time, text="Start date:").grid(row=0, column=0, padx=5, pady=5)
    entry_start_date = ttk.Entry(frame_time, width=12, textvariable=start_date_var)
    entry_start_date.grid(row=0, column=1, padx=5, pady=5)

    ttk.Button(
        frame_time,
        text="Pick",
        command=lambda: open_calendar(start_date_var, start_date_var.get() or None),
    ).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(frame_time, text="Time (HH:MM):").grid(row=0, column=3)
    entry_start_hour = ttk.Entry(frame_time, width=3)
    entry_start_min = ttk.Entry(frame_time, width=3)
    entry_start_hour.grid(row=0, column=4, padx=2)
    ttk.Label(frame_time, text=":").grid(row=0, column=5)
    entry_start_min.grid(row=0, column=6, padx=2)

    # end date/time
    ttk.Label(frame_time, text="End date:").grid(row=1, column=0, padx=5, pady=5)
    entry_end_date = ttk.Entry(frame_time, width=12, textvariable=end_date_var)
    entry_end_date.grid(row=1, column=1, padx=5, pady=5)

    ttk.Button(
        frame_time,
        text="Pick",
        command=lambda: open_calendar(end_date_var, end_date_var.get() or None),
    ).grid(row=1, column=2, padx=5, pady=5)

    ttk.Label(frame_time, text="Time (HH:MM):").grid(row=1, column=3)
    entry_end_hour = ttk.Entry(frame_time, width=3)
    entry_end_min = ttk.Entry(frame_time, width=3)
    entry_end_hour.grid(row=1, column=4, padx=2)
    ttk.Label(frame_time, text=":").grid(row=1, column=5)
    entry_end_min.grid(row=1, column=6, padx=2)

    # ==== plot button ====
    frame_btn = ttk.Frame(root)
    frame_btn.pack(pady=10)

    ttk.Button(frame_btn, text="Plot Timeseries", command=run_plot).pack()

    # ==== 4. Train model ====
    frame_train = ttk.LabelFrame(root, text="4. Train Machine Learning Model")
    frame_train.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_train, text="Train/Test split ratio (0~1):").grid(row=0, column=0, padx=5, pady=5)

    split_ratio_var = tk.StringVar(value="0.8")  # default = 80% training
    entry_split_ratio = ttk.Entry(frame_train, width=6, textvariable=split_ratio_var)
    entry_split_ratio.grid(row=0, column=1, padx=5, pady=5)

    ttk.Button(
        frame_train,
        text="Train Random Forest",
        command=lambda: on_train_random_forest(split_ratio_var, train_status_var)
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Status label for showing "progress" and metrics
    train_status_var = tk.StringVar(value="No model trained yet.")
    label_status = ttk.Label(frame_train, textvariable=train_status_var, justify="left")
    label_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    

    root.mainloop()
