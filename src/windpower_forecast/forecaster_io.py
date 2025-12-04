# 

import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path)

def mean_angle(U, phi, n):
    """
    Calculate the mean angle of a set of directions.

    Use Euler angles to get the mean angle.

    :params array U: Wind wind speed
    :params array phi: Wind directions in degrees
    :params array n: power of wind speed in normalization, U^n
    :returns: The unit vector mean direction of the wind directions
    """
    # Convert polar to complex rectangular coordinates, radius of 1
    rect_dirs = pow(U,n) * np.exp(1j * np.radians(phi))

    # Calculate mean
    rect_mean = rect_dirs.mean()

    # Convert back to polar space in degrees
    # Ensuring the result is from 0 - 359
    mean_dirs = np.mod(np.angle(rect_mean, True), 360)

    return mean_dirs


def P03_46W38_data(path):
    df = read_csv(path)
    


"""
GUI part
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar  # ✅ 用 Calendar，不用 DateEntry 了
from datetime import datetime, time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

    def list_csv_files() -> list[str]:
        """List CSV files under the provided inputs directory."""
        return sorted([p.name for p in inputs_dir.glob("*.csv")])

    def load_data(filename: str) -> pd.DataFrame:
        """Load CSV with caching."""
        if filename not in DATA_CACHE:
            df = pd.read_csv(inputs_dir / filename, parse_dates=["Time"])
            df = df.sort_values("Time").reset_index(drop=True)
            DATA_CACHE[filename] = df
        return DATA_CACHE[filename]

    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="number").columns.tolist()

    # 這兩個用來記錄目前 CSV 的時間範圍（給日曆限制用）
    time_min = None
    time_max = None

    # ---------------- Helper: 日曆選日期 ---------------- #

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

        # mindate / maxdate 限制在 CSV 範圍內（如果已載入檔案）
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
        nonlocal time_min, time_max

        filename = combo_file.get()
        if not filename:
            return

        try:
            df = load_data(filename)

            tmin = df["Time"].min()
            tmax = df["Time"].max()
            time_min, time_max = tmin, tmax

            label_range_var.set(f"Time range:\n{tmin} → {tmax}")

            # update variable dropdown
            num_cols = get_numeric_columns(df)
            combo_var["values"] = num_cols
            if num_cols:
                combo_var.set(num_cols[0])
            else:
                combo_var.set("")

            # 預設日期：整個 time range
            start_date_var.set(tmin.date().isoformat())
            end_date_var.set(tmax.date().isoformat())

            # 預設時間
            entry_start_hour.delete(0, tk.END)
            entry_start_min.delete(0, tk.END)
            entry_end_hour.delete(0, tk.END)
            entry_end_min.delete(0, tk.END)

            entry_start_hour.insert(0, "00")
            entry_start_min.insert(0, "00")
            entry_end_hour.insert(0, "23")
            entry_end_min.insert(0, "59")

        except Exception as e:
            messagebox.showerror("Error loading file", str(e))

    def run_plot():
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
            plt.figure(figsize=(10, 4))
            plt.plot(sub["Time"], sub[varname], label=varname)
            plt.xlabel("Time")
            plt.ylabel(varname)
            plt.title(f"{filename} – {varname}\n{start_dt} → {end_dt}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- Build GUI ---------------- #

    root = tk.Tk()
    root.title("Wind Power Timeseries Viewer")
    root.geometry("520x440")

    # file selection frame
    frame_file = ttk.LabelFrame(root, text="1. Choose CSV file")
    frame_file.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_file, text="CSV file:").grid(row=0, column=0, padx=5, pady=5)
    combo_file = ttk.Combobox(frame_file, width=40, state="readonly")
    combo_file["values"] = list_csv_files()
    combo_file.grid(row=0, column=1, padx=5, pady=5)
    combo_file.bind("<<ComboboxSelected>>", on_file_selected)

    label_range_var = tk.StringVar(value="Time range: (select a file)")
    ttk.Label(frame_file, textvariable=label_range_var).grid(
        row=1, column=0, columnspan=2, padx=5, pady=5
    )

    # variable selection frame
    frame_var = ttk.LabelFrame(root, text="2. Choose variable")
    frame_var.pack(fill="x", padx=10, pady=5)

    ttk.Label(frame_var, text="Variable:").grid(row=0, column=0, padx=5, pady=5)
    combo_var = ttk.Combobox(frame_var, width=30, state="readonly")
    combo_var.grid(row=0, column=1, padx=5, pady=5)

    # date selection frame
    frame_time = ttk.LabelFrame(root, text="3. Choose time period")
    frame_time.pack(fill="x", padx=10, pady=5)

    # 用 StringVar 來放日期字串
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

    # plot button
    frame_btn = ttk.Frame(root)
    frame_btn.pack(pady=10)

    ttk.Button(frame_btn, text="Plot Timeseries", command=run_plot).pack()

    root.mainloop()
