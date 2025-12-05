# GUI
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import Calendar
from datetime import datetime, time
from pathlib import Path
import threading

from .models import MODEL_TRAINERS
from . import forecaster_plot as fc_plt
from . import data

class ForecastApp:
    """
    A clean OOP-based GUI app for time-series viewing and ML forecasting.
    """

    def __init__(self, inputs_dir: Path, outputs_dir: Path | None = None):
        # store paths
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir

        # cache for loaded CSV files
        self.DATA_CACHE: dict[str, pd.DataFrame] = {}

        # time limits for calendar
        self.time_min = None
        self.time_max = None

        # last trained model & feature cols (for feature importance plot)
        self.last_model = None
        self.last_feature_cols: list[str] | None = None
        self.last_scores: dict | None = None
        self.last_model_name: str | None = None

        # model checkboxes state
        self.model_flags: dict[str, tk.BooleanVar] = {}

        # storage for predictions (for plotting)
        self.test_time: np.ndarray | None = None
        self.y_true_dict: dict[str, np.ndarray] = {}
        self.y_pred_dict: dict[str, np.ndarray] = {}

        # build GUI
        self.root = tk.Tk()
        self.root.title("Wind Power Timeseries Viewer")
        self.root.geometry("520x800")

        self.build_file_frame()
        self.build_var_frame()
        self.build_time_frame()
        self.build_plot_button()
        self.build_ml_frame()

        self.root.mainloop()

    # ---------- Data helpers ----------

    def list_csv_files(self):
        # Catch .csv files under inputs path
        return sorted([p.name for p in self.inputs_dir.glob("*.csv")])

    def load_data(self, filename: str):
        # load data 
        if filename not in self.DATA_CACHE:
            df = pd.read_csv(self.inputs_dir / filename, parse_dates=["Time"])
            df = df.sort_values("Time").reset_index(drop=True)
            self.DATA_CACHE[filename] = df
        return self.DATA_CACHE[filename]

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame):
        # Get df columns
        return df.select_dtypes(include="number").columns.tolist()

    # ---------- GUI: Calendar picker ----------

    def open_calendar(self, target_var: tk.StringVar, initial_date: str | None = None):
        top = tk.Toplevel(self.root)
        top.title("Select date")
        top.grab_set()

        # pick initial date
        if initial_date:
            try:
                init_dt = datetime.strptime(initial_date, "%Y-%m-%d").date()
            except ValueError:
                init_dt = None
        else:
            init_dt = None

        # bounds by CSV time range
        mindate = self.time_min.date() if self.time_min else None
        maxdate = self.time_max.date() if self.time_max else None

        cal_kwargs = {"selectmode": "day", "date_pattern": "yyyy-mm-dd"}
        if mindate:
            cal_kwargs["mindate"] = mindate
        if maxdate:
            cal_kwargs["maxdate"] = maxdate
        if init_dt:
            cal_kwargs["year"] = init_dt.year
            cal_kwargs["month"] = init_dt.month
            cal_kwargs["day"] = init_dt.day

        cal = Calendar(top, **cal_kwargs)
        cal.pack(padx=10, pady=10)

        def on_ok():
            target_var.set(cal.get_date())
            top.destroy()

        ttk.Button(top, text="OK", command=on_ok).pack(pady=5)

    # Folder path
    def select_folder(self):
        """Choose folder path manually, refresh CSV list"""
        new_dir = filedialog.askdirectory(
            title="Select Folder of inputs",
            initialdir=str(self.inputs_dir)
        )
        if not new_dir:
            return
        
        # Update path
        self.inputs_dir = Path(new_dir)
        self.folder_var.set(new_dir)
        self.DATA_CACHE.clear()

        # Reload
        self.combo_file["values"] = self.list_csv_files()

        if not self.list_csv_files():
            self.label_range_var.set("Time range: (no CSV files found)")
        else:
            self.label_range_var.set("Time range: ")

    # ---------- Calender GUI handlers ----------

    def on_file_selected(self, event=None):
        filename = self.combo_file.get()
        if not filename:
            return

        try:
            df = self.load_data(filename)

            tmin, tmax = df["Time"].min(), df["Time"].max()
            self.time_min, self.time_max = tmin, tmax

            self.label_range_var.set(f"Time range:\n{tmin} → {tmax}")

            # update var dropdown
            num_cols = self.get_numeric_columns(df)
            self.combo_var["values"] = num_cols
            self.combo_var.set(num_cols[0] if num_cols else "")

            # default date range
            self.start_date_var.set(tmin.date().isoformat())
            self.end_date_var.set(tmax.date().isoformat())

            # default time
            self.entry_start_hour.delete(0, tk.END)
            self.entry_start_min.delete(0, tk.END)
            self.entry_end_hour.delete(0, tk.END)
            self.entry_end_min.delete(0, tk.END)

            self.entry_start_hour.insert(0, "00")
            self.entry_start_min.insert(0, "00")
            self.entry_end_hour.insert(0, "23")
            self.entry_end_min.insert(0, "00")

        except Exception as e:
            messagebox.showerror("Error loading file", str(e))

    # --- plot ---
    def run_plot(self):
        filename = self.combo_file.get()
        varname = self.combo_var.get()

        if not filename:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        if not varname:
            messagebox.showerror("Error", "Please select a variable.")
            return

        try:
            df = self.load_data(filename)

            # get dates
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d").date()

            sh, sm = int(self.entry_start_hour.get()), int(self.entry_start_min.get())
            eh, em = int(self.entry_end_hour.get()), int(self.entry_end_min.get())

            start_dt = datetime.combine(start_date, time(sh, sm))
            end_dt = datetime.combine(end_date, time(eh, em))

            if start_dt > end_dt:
                raise ValueError("Start time cannot be after end time.")

            if start_dt < df["Time"].min() or end_dt > df["Time"].max():
                raise ValueError("Selected period is outside the CSV range.")

            mask = (df["Time"] >= start_dt) & (df["Time"] <= end_dt)
            sub = df.loc[mask]
            if sub.empty:
                raise ValueError("No data in this time window.")

            fc_plt.single_var_plot(
                sub["Time"],
                sub[varname],
                xlabel="Time",
                ylabel=varname,
                title=f"{filename} – {varname}\n{start_dt} → {end_dt}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------- ML training handlers ----------

    def on_train_button_clicked(self):
        """Validate input and start background training thread."""
        try:
            split_ratio = float(self.split_ratio_var.get())
            if not (0 < split_ratio < 1):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Split ratio must be between 0~1.")
            return
        filename = self.combo_file.get()
        if not filename:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return

        # collect models from checkboxes
        selected_models: list[str] = [
            name for name, var in self.model_flags.items() if var.get()
        ]
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model to train.")
            return
        
        # show status & start progress
        self.train_status_var.set(
            "Training models:\n  " + "\n  ".join(selected_models)
        )
        self.train_progress.start(10)
        self.btn_train.config(state="disabled")
        self.root.update_idletasks()

        # start background job
        thread = threading.Thread(
            target=self._train_model_worker,
            args=(filename, split_ratio, selected_models),
            daemon=True,
        )
        thread.start()

    def _train_model_worker(self, filename: str, split_ratio: float, model_names: list[str]):
        """Run ML training in a background thread, then hand results back to the GUI thread."""
        print(model_names)
        try:
            # load & transform
            df_raw = self.load_data(filename)
            df_ML = data.transform_features(df_raw)

            # split
            split_idx = int(len(df_ML) * split_ratio)
            train_df = df_ML.iloc[:split_idx].copy()
            test_df = df_ML.iloc[split_idx:].copy()

            # feature columns
            feature_cols = [
                "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
                "windgusts_10m",
                "u_10m", "v_10m", "u_100m", "v_100m",
                "delta_ws",
                "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            ]

            target_col = "Power_t_plus_1"

            all_scores = {}
            last_model = None

            # store test timestamps (Time kept in df_ML)
            if "Time" in test_df.columns:
                self.test_time = test_df["Time"].copy()
            else:
                self.test_time = None

            # reset stored predictions
            self.y_true_dict = {}
            self.y_pred_dict = {}

            for model_name in model_names:
                trainer_func = MODEL_TRAINERS.get(model_name)
                if trainer_func is None:
                    raise ValueError(f"Unknown model: {model_name}")

                model, scores, y_true, y_pred = trainer_func(
                    train_df, test_df, feature_cols, target_col
                )

                all_scores[model_name] = scores
                last_model = model

                # store predictions for later plotting
                self.y_true_dict[model_name] = y_true
                self.y_pred_dict[model_name] = y_pred

            # store meta info
            self.last_model = last_model
            self.last_feature_cols = feature_cols
            self.last_scores = all_scores
            self.last_model_name = ", ".join(model_names)

            # back to GUI thread
            self.root.after(0, self._on_train_done, split_ratio, all_scores)

        except Exception as e:
            self.root.after(0, self._on_train_error, str(e))


    def _on_train_done(self, split_ratio: float, all_scores: dict[str, dict]):
        """Update GUI after training finished successfully."""
        self.train_progress.stop()
        self.btn_train.config(state="normal")

        train_pct = int(split_ratio * 100)
        test_pct = 100 - train_pct

        # Begin formatted table
        lines = []
        lines.append(f"Train/Test split: {train_pct}% / {test_pct}%\n")
        lines.append(f"{'Model':22s} {'MSE':>10s} {'MAE':>10s} {'RMSE':>10s}")
        lines.append("-" * 55)

        for model_name, scores in all_scores.items():
            lines.append(
                f"{model_name:22s} "
                f"{scores['mse']:10.3f} "
                f"{scores['mae']:10.3f} "
                f"{scores['rmse']:10.3f}"
            )

        self.train_status_var.set("\n".join(lines))

    def _on_train_error(self, message: str):
        """Update GUI after training failed."""
        self.train_progress.stop()
        self.btn_train.config(state="normal")
        self.train_status_var.set("Training failed.")
        messagebox.showerror("Training error", message)

    # forecast plot
    def plot_forecast_vs_real(self):
        """
        Plot predicted one-hour-ahead power output against the real measured power
        for the selected time window and one of the trained models.

        Requirements covered:
        - Persistence model one-hour-ahead prediction
        - ML model one-hour-ahead prediction (SVM, Random Forest, MLP, etc.)
        - Plot predicted vs real time series for a selected site and period
          (GUI start/end date & time are used as the period)
        """

        # Check if any model has been trained
        if self.last_scores is None or not self.y_true_dict:
            messagebox.showerror("Error", "Please train at least one model first.")
            return

        if self.test_time is None:
            messagebox.showerror("Error", "No stored test timestamps available.")
            return

        time_arr = self.test_time

        # Prefer SVM if it was trained; otherwise take the first available model
        preferred = "Support Vector Machine"
        if preferred in self.y_true_dict:
            model_name = preferred
        else:
            model_name = next(iter(self.y_true_dict.keys()))

        y_true = self.y_true_dict[model_name]
        y_pred = self.y_pred_dict[model_name]

        # Convert GUI start/end date + time into real datetimes
        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d").date()

            sh, sm = int(self.entry_start_hour.get()), int(self.entry_start_min.get())
            eh, em = int(self.entry_end_hour.get()), int(self.entry_end_min.get())

            start_dt = datetime.combine(start_date, time(sh, sm))
            end_dt = datetime.combine(end_date, time(eh, em))

            if start_dt > end_dt:
                raise ValueError("Start time cannot be after end time.")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid date/time: {e}")
            return

        # Slice the period selected from GUI
        mask = (time_arr >= start_dt) & (time_arr <= end_dt)
        if not mask.any():
            messagebox.showerror("Error", "No test data in the selected time window.")
            return

        # Call plotting function
        fc_plt.forecast_vs_real_plot(
            time_arr[mask],
            y_true[mask],
            y_pred[mask],
            title=f"{model_name} Forecast vs Real\n{start_dt} → {end_dt}",
        )



    # ---------- GUI Builders ----------

    def build_file_frame(self):
        frame = ttk.LabelFrame(self.root, text="1. Choose CSV file")
        frame.pack(fill="x", padx=10, pady=5)

        # Folder selector
        ttk.Label(frame, text="Folder:").grid(row=0, column=0, padx=5, pady=5)
        self.folder_var = tk.StringVar(value=str(self.inputs_dir))
        folder_entry = ttk.Entry(frame, textvariable=self.folder_var, width=35)
        folder_entry.grid(row=0, column=1, padx=5, pady=5)
        #Browse folder button
        ttk.Button(frame, text="...", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)

        # CSV selector
        ttk.Label(frame, text="CSV file:").grid(row=1, column=0, padx=5, pady=5)

        self.combo_file = ttk.Combobox(frame, width=40, state="readonly")
        self.combo_file["values"] = self.list_csv_files()
        self.combo_file.grid(row=1, column=1, padx=5, pady=5)
        self.combo_file.bind("<<ComboboxSelected>>", self.on_file_selected)

        self.label_range_var = tk.StringVar(value="Time range: (select a file)")
        ttk.Label(frame, textvariable=self.label_range_var).grid(
            row=2, column=0, columnspan=3, padx=5, pady=5
        )

    def build_var_frame(self):
        frame = ttk.LabelFrame(self.root, text="2. Choose variable")
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame, text="Variable:").grid(row=0, column=0, padx=5, pady=5)
        self.combo_var = ttk.Combobox(frame, width=30, state="readonly")
        self.combo_var.grid(row=0, column=1, padx=5, pady=5)

    def build_time_frame(self):
        frame = ttk.LabelFrame(self.root, text="3. Choose time period")
        frame.pack(fill="x", padx=10, pady=5)

        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()

        # start date
        ttk.Label(frame, text="Start date:").grid(row=0, column=0, padx=5, pady=5)
        entry = ttk.Entry(frame, width=12, textvariable=self.start_date_var)
        entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(
            frame, text="Pick",
            command=lambda: self.open_calendar(self.start_date_var, self.start_date_var.get())
        ).grid(row=0, column=2, padx=5)

        self.entry_start_hour = ttk.Entry(frame, width=3)
        self.entry_start_min = ttk.Entry(frame, width=3)
        ttk.Label(frame, text="Time (HH:MM):").grid(row=0, column=3, padx=5)
        self.entry_start_hour.grid(row=0, column=4)
        ttk.Label(frame, text=":").grid(row=0, column=5)
        self.entry_start_min.grid(row=0, column=6)

        # end date
        ttk.Label(frame, text="End date:").grid(row=1, column=0, padx=5, pady=5)
        entry = ttk.Entry(frame, width=12, textvariable=self.end_date_var)
        entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(
            frame, text="Pick",
            command=lambda: self.open_calendar(self.end_date_var, self.end_date_var.get())
        ).grid(row=1, column=2, padx=5)

        self.entry_end_hour = ttk.Entry(frame, width=3)
        self.entry_end_min = ttk.Entry(frame, width=3)
        ttk.Label(frame, text="Time (HH:MM):").grid(row=1, column=3, padx=5)
        self.entry_end_hour.grid(row=1, column=4)
        ttk.Label(frame, text=":").grid(row=1, column=5)
        self.entry_end_min.grid(row=1, column=6)

    def build_plot_button(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Plot Timeseries", command=self.run_plot).pack()

    def build_ml_frame(self):
        frame = ttk.LabelFrame(self.root, text="4. Train Machine Learning Model")
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame, text="Train/Test split ratio (0~1):").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.split_ratio_var = tk.StringVar(value="0.8")
        ttk.Entry(frame, width=6, textvariable=self.split_ratio_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Model selection
        ttk.Label(frame, text="Models to train:").grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        
        chk_frame = ttk.Frame(frame)
        chk_frame.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.model_flags = {}
        
        for i, model_name in enumerate(MODEL_TRAINERS.keys()):
            var = tk.BooleanVar()
            self.model_flags[model_name] = var
            ttk.Checkbutton(
                chk_frame,
                text=model_name,
                variable=var,
            ).grid(row=i, column=0, sticky="w")

        # Train button + progress bar
        self.btn_train = ttk.Button(
            frame, text="Train selected models",
            command=self.on_train_button_clicked,
        )
        self.btn_train.grid(row=3, column=0, columnspan=2, pady=10)

        # bar
        self.train_progress = ttk.Progressbar(frame, mode="indeterminate")
        self.train_progress.grid(row=4, column=0, columnspan=2, sticky="we")

        # Status text
        self.train_status_var = tk.StringVar(value="No model trained yet.")
        ttk.Label(frame, textvariable=self.train_status_var, justify="left", font=("Courier New", 10),
        ).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # plot forecast
        ttk.Button(
            frame,
            text="Plot forecast vs real",
            command=self.plot_forecast_vs_real
        ).grid(row=6, column=0, columnspan=2, pady=10)


# ------- Public function called by main.py -------
def launch_gui(inputs_dir: Path, outputs_dir: Path | None = None):
    ForecastApp(inputs_dir, outputs_dir)
