"""Laser Beam Profiler - live 1/e^2 widths AND Gaussian fit plots.
Real-time X/Y intensity profile plots appear in a fixed two-column analytics area.

Place this file next to Beam_ROI_Finder.py, Beam_Center_Finder.py, Beam_W_Finder.py.
Run: python Laser_Profiler_GUI_Spatial_Calibration.py

The camera preview is ABOVE the heatmap. Both use native-resolution
frames for analysis; frames are resized only for GUI display.
"""

import contextlib
import io
import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from Beam_ROI_Finder import find_recursive_roi
from Beam_Center_Finder import refine_center
from Beam_W_Finder import measure_beam_width
from Beam_Gaussian_Fit import fit_gaussian_profile
from Beam_Spatial_Calibration import (
    DIAGONAL_FOV_DEGREES, geometric_mm_per_pixel,
)


CAMERA_INDEX = 0
BACKGROUND_FRAMES = 30
ROI_GRID_SIZES = [8, 4]  # Peak-constrained ROI; two iterations
STABILITY_SAMPLES = 100
WIDTH_HALF_SIZE = 50  # A 100x100 px analysis window, centered on the detected spot
MIN_RELIABLE_RADIUS_PX = 5.0  # Advisory only, not a calibrated accuracy guarantee
PROFILE_UPDATE_INTERVAL = 0.20  # Fit + graph update up to 5 Hz; tracking remains independent
FIT_R2_WARNING = 0.97  # Exploratory fit diagnostic; NOT a metrology acceptance criterion
FIT_WIDTH_MISMATCH_WARNING = 0.20  # Flag >20% disagreement between width estimators
# Files are resolved relative to this project, not a user-specific Windows path.
# For a packaged EXE, outputs are stored alongside the EXE (in writable folders).
PROJECT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))
OUTPUT_DIR = PROJECT_DIR / "Snapshots"
WIDTH_OUTPUT_DIR = PROJECT_DIR / "W Results"


BG = "#111827"
PANEL = "#1f2937"
TEXT = "#f9fafb"
MUTED = "#aab5c5"
ACCENT = "#5eead4"
TITLE_COLOR = "#E2E8F0"
DIVIDER_COLOR = "#435166"


class IntensityProfilePlot(tk.Canvas):
    """Small, lightweight profile graph rendered with native Tk Canvas.

    Keeps video/analysis separate from chart refresh; matplotlib is not needed.
    The signal is the camera-derived, background-subtracted channel difference,
    NOT calibrated optical irradiance.
    """

    def __init__(self, parent, axis_name):
        super().__init__(
            parent, width=285, height=108, bg="#111827", highlightthickness=1,
            highlightbackground="#374151", bd=0,
        )
        self.axis_name = axis_name
        self.profile_data = None
        self.radius = None
        self.fit_data = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.redraw()

    def set_profile(self, profile_data, radius=None, gaussian_fit=None):
        self.profile_data = profile_data
        self.radius = radius
        self.fit_data = gaussian_fit
        self.redraw()

    def redraw(self):
        self.delete("all")
        width = max(150, self.winfo_width())
        height = max(120, self.winfo_height())
        l, r = 35, width - 11
        t, b = 23, height - 29

        self.create_text(
            width / 2, 5,
            text=f"{self.axis_name} PROFILE",
            anchor="n",
             fill=TITLE_COLOR,
             font=("Segoe UI", 11, "bold")
        )

        data = self.profile_data
        if data is None:
            self.create_text(width / 2, height / 2, text="No valid width profile",
                             fill="#94a3b8", font=("Segoe UI", 10))
            return

        x = np.asarray(data.get("axis", []), dtype=float)
        y = np.asarray(data.get("profile", []), dtype=float)
        if x.size < 2 or y.size != x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
            self.create_text(width / 2, height / 2, text="Profile unavailable", fill="#94a3b8")
            return

        x_min, x_max = float(x[0]), float(x[-1])
        if x_max <= x_min:
            return
        # Preserve negative background residuals in the graph instead of hiding them.
        y_min = min(-0.05, float(y.min()) - 0.03)
        y_max = max(1.05, float(y.max()) + 0.03)
        threshold = float(np.exp(-2))

        def px(value):
            return l + (float(value) - x_min) * (r - l) / (x_max - x_min)

        def py(value):
            return b - (float(value) - y_min) * (b - t) / (y_max - y_min)

        # Reference grid and labels
        for level, caption in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            line_y = py(level)
            self.create_line(l, line_y, r, line_y, fill="#334155", dash=(2, 3))
            self.create_text(l - 5, line_y, text=caption, anchor="e", fill="#94a3b8",
                             font=("Segoe UI", 10))
        self.create_line(l, t, l, b, fill="#64748b")
        self.create_line(l, b, r, b, fill="#64748b")
        for position in (x_min, (x_min + x_max) / 2, x_max):
            self.create_text(px(position), b + 5, text=f"{position:.0f}",
                             anchor="n", fill="#94a3b8", font=("Segoe UI", 10))

        # 1/e^2 reference level + crossing markers, from the actual width finder
        self.create_line(l, py(threshold), r, py(threshold),
                         fill="#fbbf24", width=1, dash=(5, 3))
        self.create_text(r - 2, py(threshold) - 4, text="1/e²",
                         anchor="se", fill="#fbbf24", font=("Segoe UI", 10))
        for crossing_key in ("left", "right"):
            crossing = data.get(crossing_key)
            if crossing is not None and np.isfinite(crossing) and x_min <= crossing <= x_max:
                cross_x = px(crossing)
                self.create_line(cross_x, t, cross_x, b,
                                 fill="#fb923c", width=1, dash=(3, 4))
                rad = 3
                self.create_oval(cross_x - rad, py(threshold) - rad,
                                 cross_x + rad, py(threshold) + rad,
                                 fill="#fb923c", outline="")

        # Measured (smoothed, normalized) intensity curve
        coords = []
        for xi, yi in zip(x, y):
            coords.extend((px(xi), py(yi)))
        self.create_line(*coords, fill="#38bdf8", width=2)

        # A Gaussian fit is shown independently of the measured 1/e^2 crossings.
        fit = self.fit_data
        if fit is not None:
            fit_y = np.asarray(fit.get("fitted_profile", []), dtype=float)
            if fit_y.size == x.size and np.isfinite(fit_y).all():
                fit_coords = []
                for xi, yi in zip(x, fit_y):
                    fit_coords.extend((px(xi), py(yi)))
                self.create_line(*fit_coords, fill="#a78bfa", width=2,
                                 dash=(6, 3))
                self.create_text(r, t + 4,
                                 text=f"Gaussian w = {fit['radius']:.1f} | R² = {fit['r_squared']:.3f}",
                                 anchor="ne", fill="#c4b5fd", font=("Segoe UI", 10))

        if self.radius is not None and np.isfinite(self.radius):
            self.create_text(r, 6, text=f"w = {self.radius:.2f} px", anchor="ne",
                             fill="#7dd3fc", font=("Segoe UI", 11, "bold"))


class LaserProfilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Beam Tracker and Profiler - Itamar Aloni")
        self.root.configure(bg=BG)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{min(1710, screen_width - 32)}x{min(980, screen_height - 70)}")
        root.minsize(1090, 650)

        self.cap = None
        self.last_frame = None
        self.heatmap = None
        self.background = None
        self.noise_level = 1.0
        self.calibrating = False
        self.calibration_count = 0
        self.calibration_mean = None
        self.calibration_m2 = None
        self.tracking_enabled = True
        self.last_analysis_started = -float("inf")
        self.analysis_timestamps = deque(maxlen=40)
        self.center = None
        self.roi_bounds = None
        self.stability_active = False
        self.stability_positions = []
        self.width_result = None
        self.width_crop_bounds = None
        self.width_warning = ""
        self.width_saturated_pixels = 0
        self.gaussian_fits = {}
        self._last_profile_refresh = -float("inf")
        self._profile_dirty = True
        self.running = True

        self.resolution_var = tk.StringVar(value="1280 x 720")
        self.rate_var = tk.StringVar(value="20 Hz")
        self.status_var = tk.StringVar(value="Opening camera...")
        self.camera_info_var = tk.StringVar(value="Camera: --")
        self.x_var = tk.StringVar(value="-- px")
        self.y_var = tk.StringVar(value="-- px")
        self.analysis_ms_var = tk.StringVar(value="-- ms")
        self.fps_var = tk.StringVar(value="-- Hz")
        self.saturation_var = tk.StringVar(value="--")
        self.wx_var = tk.StringVar(value="-- px")
        self.wy_var = tk.StringVar(value="-- px")
        self.gwx_var = tk.StringVar(value="-- px")
        self.gwy_var = tk.StringVar(value="-- px")
        self.fit_status_var = tk.StringVar(value="No Gaussian fit yet")
        self.width_status_var = tk.StringVar(value="Calibrate camera first")
        self.stability_var = tk.StringVar(value="Not running")

        # Spatial estimates are optional and never modify the measured px values.
        self.distance_entry_var = tk.StringVar(value="30")
        self.distance_cm = None  # Disabled until Apply is clicked.
        self.scale_mm_per_px = None
        self.spatial_scale_var = tk.StringVar(value="Enter distance and click Apply (720p)")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-b>", lambda _: self.start_calibration())
        self.root.bind("<KeyPress-q>", lambda _: self.close())

        self.open_camera()
        self.root.after(25, self.update)

    def _preview_title(self, parent, row, text, top=0):
        header = tk.Frame(parent, bg=BG)
        header.grid(
            row=row, column=0,
            sticky="ew",
            pady=(top, 5)
        )

        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text=text,
            bg=BG,
            fg=TITLE_COLOR,
            font=("Segoe UI Semibold", 12, "bold"),
            anchor="center",
            justify="center"
        ).grid(row=0, column=0, sticky="ew")

    def _build_ui(self):
        """Compact no-scroll dashboard: left 16:9 images, center large live
        profiles + metrics + widths, right all controls/calibration/photo.
        Detection and numerical analysis continue on the native camera frame.
        """
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", padding=(4, 2), font=("Segoe UI", 10))
        style.configure("TCombobox", padding=1, font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))

        heading = tk.Frame(self.root, bg=BG)
        heading.pack(fill="x", padx=12, pady=(9, 9))
        tk.Label(
            heading, text="LASER BEAM TRACKER AND PROFILER - ITAMAR ALONI",
            font=("Segoe UI Semibold", 20, "bold"), bg=BG, fg=TEXT,
        ).pack(anchor="center")

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=10, pady=(0, 9))
        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=0)
        content.grid_rowconfigure(0, weight=1)
        self._content = content

        # LEFT: native-ratio camera and heatmap, vertically stacked.
        left = tk.Frame(content, bg=BG, width=540)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 9))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        self._preview_left = left

        self._preview_title(left, 0, "LIVE CAMERA")

        camera_box = tk.Frame(left, bg="#05070c", width=540, height=303,
                              highlightbackground="#364152", highlightthickness=1)
        camera_box.grid(row=1, column=0, sticky="w")
        camera_box.pack_propagate(False)
        self.camera_label = tk.Label(camera_box, bg="#05070c", bd=0)
        self.camera_label.pack(fill="both", expand=True)
        self._camera_box = camera_box

        self._preview_title(
            left, 2, "BACKGROUND-SUBTRACTED HEATMAP", top=10
        )

        heat_box = tk.Frame(left, bg="#05070c", width=540, height=303,
                            highlightbackground="#364152", highlightthickness=1)
        heat_box.grid(row=3, column=0, sticky="w")
        heat_box.pack_propagate(False)
        self.heatmap_label = tk.Label(heat_box, bg="#05070c", bd=0)
        self.heatmap_label.pack(fill="both", expand=True)
        self._heat_box = heat_box

        # CENTER: metrics, largest possible live graphs, then beam widths.
        analysis = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=DIVIDER_COLOR,
            highlightthickness=1
        )
        analysis.grid(row=0, column=1, sticky="nsew", padx=(0, 9))
        analysis.grid_columnconfigure(0, weight=1)
        self.analysis_controls = analysis

        r = 0
        self._section_title(analysis, r, "LIVE MEASUREMENTS")
        r += 1
        metrics = tk.Frame(analysis, bg=PANEL)
        metrics.grid(row=r, column=0, sticky="ew", padx=10, pady=(2, 5))
        for i in range(6):
            metrics.grid_columnconfigure(i, weight=1, uniform="metrics")
        for col, span, text, var in (
            (0, 3, "CENTER X", self.x_var),
            (3, 3, "CENTER Y", self.y_var),
        ):
            self._metric_tile(metrics, 0, col, span, text, var)
        for col, span, text, var in (
            (0, 2, "ANALYSIS TIME", self.analysis_ms_var),
            (2, 2, "ACTUAL RATE", self.fps_var),
            (4, 2, "SATURATED PIXELS", self.saturation_var),
        ):
            self._metric_tile(metrics, 1, col, span, text, var)

        r += 1
        self._section_title(analysis, r, "LIVE INTENSITY PROFILES")
        r += 1
        tk.Label(analysis, text="Measured signal  /  Gaussian fit  /  1/e² crossings  ·  up to 5 Hz",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).grid(
            row=r, column=0, sticky="w", padx=12, pady=(0, 4))
        r += 1
        self.x_profile_plot = IntensityProfilePlot(analysis, "X")
        self.x_profile_plot.grid(row=r, column=0, sticky="nsew", padx=12, pady=(0, 6))
        analysis.grid_rowconfigure(r, weight=1, uniform="graphs")
        r += 1
        self.y_profile_plot = IntensityProfilePlot(analysis, "Y")
        self.y_profile_plot.grid(row=r, column=0, sticky="nsew", padx=12, pady=(0, 7))
        analysis.grid_rowconfigure(r, weight=1, uniform="graphs")

        r += 1
        self._section_title(analysis, r, "BEAM WIDTH (EXPERIMENTAL)")
        r += 1
        width_data = tk.Frame(analysis, bg=PANEL)
        width_data.grid(row=r, column=0, sticky="ew", padx=10, pady=(1, 3))
        width_data.grid_columnconfigure(0, weight=1)
        width_data.grid_columnconfigure(1, weight=1)
        for rr, cc, label, var in (
            (0, 0, "1/e² radius X", self.wx_var),
            (0, 1, "1/e² radius Y", self.wy_var),
            (1, 0, "Gaussian fit X", self.gwx_var),
            (1, 1, "Gaussian fit Y", self.gwy_var),
        ):
            frame = tk.Frame(width_data, bg="#17243a")
            frame.grid(row=rr, column=cc, sticky="ew", padx=3, pady=3)
            tk.Label(frame, text=label, bg="#17243a", fg=MUTED,
                     font=("Segoe UI", 10)).pack(anchor="w", padx=9, pady=(2, 0))
            tk.Label(frame, textvariable=var, bg="#17243a", fg=TEXT,
                     font=("Segoe UI Semibold", 12, "bold")).pack(anchor="w", padx=9, pady=(0, 2))

        # RIGHT: camera controls, scale, stability, status, personal photo.
        controls = tk.Frame(
            content,
            bg=PANEL,
            width=380,
            highlightbackground=DIVIDER_COLOR,
            highlightthickness=1
        )
        controls.grid(row=0, column=2, sticky="nsew")
        controls.grid_propagate(False)
        controls.grid_columnconfigure(0, weight=1)
        self.controls = controls
        r = 0
        self._section_title(controls, r, "CAMERA CONTROLS")
        r += 1
        res_row = tk.Frame(controls, bg=PANEL)
        res_row.grid(row=r, column=0, sticky="ew", padx=10, pady=1)
        tk.Label(res_row, text="Resolution", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), width=12, anchor="w").pack(side="left")
        res = ttk.Combobox(res_row, state="readonly", textvariable=self.resolution_var,
                           values=["1280 x 720", "640 x 480"], width=13)
        res.pack(side="right", fill="x", expand=True)
        res.bind("<<ComboboxSelected>>", lambda _: self.open_camera())
        r += 1
        rate_row = tk.Frame(controls, bg=PANEL)
        rate_row.grid(row=r, column=0, sticky="ew", padx=10, pady=1)
        tk.Label(rate_row, text="Tracking rate", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), width=12, anchor="w").pack(side="left")
        ttk.Combobox(rate_row, state="readonly", textvariable=self.rate_var,
                     values=["5 Hz", "10 Hz", "15 Hz", "20 Hz", "30 Hz", "Maximum"],
                     width=13).pack(side="right", fill="x", expand=True)
        r += 1
        button_row = tk.Frame(controls, bg=PANEL)
        button_row.grid(row=r, column=0, sticky="ew", padx=10, pady=(3, 1))
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Camera Settings", command=self.open_camera_settings).grid(
            row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(button_row, text="Calibrate Background", command=self.start_calibration).grid(
            row=0, column=1, sticky="ew", padx=(3, 0))
        r += 1
        button_row = tk.Frame(controls, bg=PANEL)
        button_row.grid(row=r, column=0, sticky="ew", padx=10, pady=1)
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)
        self.toggle_button = ttk.Button(button_row, text="Pause Tracking", command=self.toggle_tracking)
        self.toggle_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(button_row, text="Save Snapshot", command=self.save_snapshot).grid(
            row=0, column=1, sticky="ew", padx=(3, 0))
        r += 1
        ttk.Button(controls, text="Save Width Profiles (CSV)", command=self.save_width_profiles).grid(
            row=r, column=0, sticky="ew", padx=10, pady=(1, 2))

        r += 1
        self._section_title(controls, r, "SPATIAL CALIBRATION")
        r += 1
        distance_row = tk.Frame(controls, bg=PANEL)
        distance_row.grid(row=r, column=0, sticky="ew", padx=10, pady=(2, 2))
        tk.Label(distance_row, text="Wall (cm)", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        self.distance_entry = ttk.Entry(distance_row, textvariable=self.distance_entry_var, width=7)
        self.distance_entry.pack(side="left", padx=(8, 5))
        ttk.Button(distance_row, text="Apply", width=6,
                   command=self.apply_spatial_calibration).pack(side="left")
        ttk.Button(distance_row, text="Clear", width=6,
                   command=self.clear_spatial_calibration).pack(side="left", padx=(5, 0))
        r += 1
        tk.Label(controls, textvariable=self.spatial_scale_var, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), anchor="w", justify="left",
                 wraplength=360).grid(row=r, column=0, sticky="ew", padx=12, pady=0)
        r += 1

        self._section_title(controls, r, "STABILITY TEST")
        r += 1
        ttk.Button(controls, text="Collect 100 Positions", command=self.start_stability).grid(
            row=r, column=0, sticky="ew", padx=10, pady=1)
        r += 1
        tk.Label(controls, textvariable=self.stability_var, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), justify="left", wraplength=350).grid(
            row=r, column=0, sticky="ew", padx=12, pady=1)
        r += 1
        self._section_title(controls, r, "STATUS")
        r += 1
        tk.Label(controls, textvariable=self.status_var, bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Semibold", 10, "bold"), justify="left",
                 wraplength=350).grid(row=r, column=0, sticky="ew", padx=12, pady=(2, 1))
        r += 1
        tk.Label(controls, textvariable=self.camera_info_var, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), justify="left").grid(
            row=r, column=0, sticky="ew", padx=12, pady=1)
        r += 1
        tk.Label(controls, text="B: calibrate  |  Q: quit   ·   Laser OFF during calibration",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 wraplength=350, justify="left").grid(
            row=r, column=0, sticky="ew", padx=12, pady=(0, 4))

        # The photo fills the remaining lower-right area (cropped, not distorted).
        r += 1
        controls.grid_rowconfigure(r, weight=1)
        self.photo_canvas = tk.Canvas(controls, bg=PANEL, highlightthickness=0, bd=0)
        self.photo_canvas.grid(row=r, column=0, sticky="nsew", padx=9, pady=(2, 6))
        self._photo_image_item = self.photo_canvas.create_image(0, 0, anchor="nw")
        self.status_original = None
        for photo_path in (
            PROJECT_DIR / "status_photo.jpg", RESOURCE_DIR / "status_photo.jpg"
        ):
            if photo_path.exists():
                self.status_original = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
                break
        if self.status_original is None:
            self.photo_canvas.create_text(20, 22, anchor="nw", text="status_photo.jpg not found",
                                          fill=MUTED, font=("Segoe UI", 10))
        else:
            self.photo_canvas.bind("<Configure>", self._resize_status_photo)

        content.bind("<Configure>", self._on_content_resize)

    def _metric_tile(self, parent, row, column, colspan, title, variable):
        panel = tk.Frame(parent, bg="#17243a")
        panel.grid(row=row, column=column, columnspan=colspan, sticky="ew", padx=3, pady=3)
        tk.Label(panel, text=title, bg="#17243a", fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=9, pady=(2, 0))
        tk.Label(panel, textvariable=variable, bg="#17243a", fg=TEXT,
                 font=("Segoe UI Semibold", 12, "bold")).pack(anchor="w", padx=9, pady=(0, 2))

    
    def _section_title(self, parent, row, text):
        header = tk.Frame(parent, bg=PANEL)
        header.grid(
            row=row,
            column=0,
            sticky="ew",
            padx=12,
            pady=(7, 4)
        )

        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text=text,
            bg=PANEL,
            fg=TITLE_COLOR,
            font=("Segoe UI Semibold", 12, "bold"),
            anchor="center",
            justify="center"
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 5)
        )

        tk.Frame(
            header,
            bg=DIVIDER_COLOR,
            height=1
        ).grid(
            row=1,
            column=0,
            sticky="ew"
        )


    def _resize_status_photo(self, event=None):
        if self.status_original is None:
            return
        width = self.photo_canvas.winfo_width()
        height = self.photo_canvas.winfo_height()
        if width < 20 or height < 20:
            return
        if getattr(self, "_status_photo_size", None) == (width, height):
            return
        self._status_photo_size = (width, height)
        resized = ImageOps.fit(self.status_original, (width, height),
                               method=Image.Resampling.LANCZOS,
                               centering=(0.5, 0.45))
        self.status_photo = ImageTk.PhotoImage(resized)
        self.photo_canvas.itemconfigure(self._photo_image_item, image=self.status_photo)

    def _on_content_resize(self, event=None):
        """No letterboxing: set each PREVIEW frame to the source image aspect.

        Use remaining horizontal space for the large central X/Y plots and right
        controls; the bottom-right photo grows to absorb leftover vertical space.
        """
        width = event.width if event is not None else self._content.winfo_width()
        height = event.height if event is not None else self._content.winfo_height()
        if width < 400 or height < 340:
            return

        if self.last_frame is None:
            aspect = 16 / 9
        else:
            source_h, source_w = self.last_frame.shape[:2]
            aspect = source_w / max(1, source_h)

        # Limit side panel to 25-27% and give the rest to graphs/preview.
        right_width = max(300, min(430, int(width * 0.245)))
        desired_preview_width = int((height - 66) * aspect / 2)
        preview_width = max(270, min(desired_preview_width, width - right_width - 400))
        preview_height = max(90, int(round(preview_width / aspect)))
        preview_size = (preview_width, preview_height, right_width, height)
        if getattr(self, "_preview_layout", None) != preview_size:
            self._preview_layout = preview_size
            self._preview_left.configure(width=preview_width + 1, height=height)
            self._camera_box.configure(width=preview_width, height=preview_height)
            self._heat_box.configure(width=preview_width, height=preview_height)
            self.controls.configure(width=right_width, height=height)
        graph_width = max(240, width - right_width - preview_width - 45)

    def apply_spatial_calibration(self):
        """Activate distance-based FOV conversion (not a measured calibration)."""
        raw = self.distance_entry_var.get().strip().replace(",", ".")
        try:
            distance = float(raw)
            if not np.isfinite(distance) or not (0 < distance < 100000):
                raise ValueError("Invalid distance")
        except ValueError:
            messagebox.showerror("Spatial Calibration", "Enter a positive distance in cm (e.g. 50).")
            return

        self.distance_cm = distance
        self._recalculate_spatial_scale()

    def clear_spatial_calibration(self):
        self.distance_cm = None
        self.scale_mm_per_px = None
        self._update_spatial_values()

    def _recalculate_spatial_scale(self):
        """Require native 720p to avoid assuming 480p is the same FOV/crop."""
        self.scale_mm_per_px = None
        if self.distance_cm is not None and self.last_frame is not None:
            h, w = self.last_frame.shape[:2]
            try:
                self.scale_mm_per_px = geometric_mm_per_pixel(self.distance_cm, w, h)
            except ValueError:
                pass
        self._update_spatial_values()

    def _update_spatial_values(self):
        """Display estimated spot radii as px / mm; leave numerical analysis intact."""
        scale = self.scale_mm_per_px
        if self.distance_cm is None:
            self.spatial_scale_var.set("Enter distance + Apply | FOV 68.5° (720p)")
        elif scale is None:
            self.spatial_scale_var.set("Scale unavailable: requires native 1280x720")
        else:
            note = " | outside focus range" if not 30 <= self.distance_cm <= 150 else ""
            self.spatial_scale_var.set(
                f"Approx. {scale:.4f} mm/px | FOV {DIAGONAL_FOV_DEGREES:.1f}°{note}"
            )

        def show_radius(value, r2=None):
            if value is None:
                result = "-- px / -- mm"
            else:
                value = float(value)
                result = f"{value:.2f} px / {value * scale:.2f} mm" if scale is not None \
                    else f"{value:.2f} px / -- mm"
            if r2 is not None:
                result += f" | R² {float(r2):.3f}"
            return result

        if self.width_result is None:
            self.wx_var.set(show_radius(None))
            self.wy_var.set(show_radius(None))
        else:
            self.wx_var.set(show_radius(self.width_result['wx']))
            self.wy_var.set(show_radius(self.width_result['wy']))

        for axis, output_var in (("x", self.gwx_var), ("y", self.gwy_var)):
            fit = self.gaussian_fits.get(axis)
            if self.width_result is None or fit is None:
                output_var.set(show_radius(None))
            else:
                output_var.set(show_radius(fit['radius'], fit['r_squared']))

    def invalidate_background(self):
        self.background = None
        self.calibrating = False
        self.calibration_count = 0
        self.calibration_mean = None
        self.calibration_m2 = None
        self.noise_level = 1.0
        self.center = None
        self.roi_bounds = None
        self.stability_active = False
        self.stability_positions = []
        self.stability_var.set("Not running")
        self.analysis_timestamps.clear()
        self.last_analysis_started = -float("inf")
        self.x_var.set("-- px")
        self.y_var.set("-- px")
        self.fps_var.set("-- Hz")
        self.analysis_ms_var.set("-- ms")
        self.saturation_var.set("--")
        self.width_result = None
        self.width_crop_bounds = None
        self.width_warning = ""
        self.width_saturated_pixels = 0
        self.gaussian_fits = {}
        self.wx_var.set("-- px")
        self.gwx_var.set("-- px")
        self.gwy_var.set("-- px")
        self.fit_status_var.set("No Gaussian fit yet")
        self.wy_var.set("-- px")
        self.width_status_var.set("Calibrate camera first")
        self._profile_dirty = True
        self.refresh_profile_charts()
        self._update_spatial_values()

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            width, height = [int(s.strip()) for s in self.resolution_var.get().split("x")]
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera. Check that no other app is using it.")

            # Ask for MJPEG before requesting the resolution (helps 720p on some webcams).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError("Camera opened, but no image was returned.")

            self.cap = cap
            self.last_frame = frame
            self.heatmap = np.zeros_like(frame)
            actual_h, actual_w = frame.shape[:2]
            self.invalidate_background()
            self.camera_info_var.set(f"Actual camera: {actual_w} x {actual_h}")
            self._recalculate_spatial_scale()
            self.status_var.set("Laser OFF -> Calibrate Background")
            # Recalculate the preview aspect ratio after switching between
            # 720p (16:9) and 480p (4:3), even if the window is unchanged.
            self.root.after_idle(self._on_content_resize)
        except Exception as exc:
            self.status_var.set("Camera error")
            messagebox.showerror("Camera", str(exc))

    def open_camera_settings(self):
        if self.cap is None:
            return
        # Windows DirectShow property pages; not all camera drivers support them.
        if not self.cap.set(cv2.CAP_PROP_SETTINGS, 1):
            messagebox.showinfo("Camera Settings", "The driver did not expose camera settings here.")
        self.invalidate_background()
        self.status_var.set("Camera settings checked -> recalibrate (B)")

    def start_calibration(self):
        if self.cap is None:
            return
        self.invalidate_background()
        self.calibrating = True
        self.status_var.set("Calibrating 0/30 - LASER OFF")

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        self.toggle_button.configure(
            text="Pause Tracking" if self.tracking_enabled else "Resume Tracking"
        )
        self.status_var.set("Tracking resumed" if self.tracking_enabled else "Tracking paused")
        if not self.tracking_enabled:
            self.center = None
            self.roi_bounds = None
            self.width_result = None
            self.width_crop_bounds = None
            self.width_saturated_pixels = 0
            self.wx_var.set("-- px")
            self.wy_var.set("-- px")
            self.gwx_var.set("-- px")
            self.gwy_var.set("-- px")
            self.fit_status_var.set("Tracking paused")
            self.width_status_var.set("Tracking paused")
            self._profile_dirty = True
            self.refresh_profile_charts()
            self._update_spatial_values()

    def start_stability(self):
        if self.background is None:
            messagebox.showinfo("Stability", "Calibrate the background first, with the laser OFF.")
            return
        if not self.tracking_enabled:
            messagebox.showinfo("Stability", "Resume tracking before starting the test.")
            return
        self.stability_active = True
        self.stability_positions = []
        self.stability_var.set("Collecting: 0/100")

    def process_calibration_frame(self, frame):
        sample = frame.astype(np.float32)
        self.calibration_count += 1
        n = self.calibration_count
        if n == 1:
            self.calibration_mean = sample.copy()
            self.calibration_m2 = np.zeros_like(sample)
        else:
            # Online mean/variance: no 30-frame stack needed at 720p.
            delta = sample - self.calibration_mean
            self.calibration_mean += delta / n
            delta2 = sample - self.calibration_mean
            self.calibration_m2 += delta * delta2

        self.status_var.set(f"Calibrating {n}/{BACKGROUND_FRAMES} - LASER OFF")
        if n >= BACKGROUND_FRAMES:
            self.background = self.calibration_mean.copy()
            standard_deviation = np.sqrt(np.maximum(self.calibration_m2 / n, 0))
            noise_map = np.max(standard_deviation, axis=2)
            self.noise_level = max(1.0, float(np.percentile(noise_map, 90)))
            self.calibrating = False
            self.calibration_mean = None
            self.calibration_m2 = None
            self.status_var.set("Calibrated - turn laser ON")

    @staticmethod
    def extract_signal(frame, background):
        # The largest positive B/G/R change: wavelength-agnostic localization,
        # NOT a linear or radiometrically calibrated optical intensity.
        return np.max(np.maximum(frame.astype(np.float32) - background, 0), axis=2)

    def measure_width_for_frame(self, frame, signal, xc, yc):
        """Estimate apparent 1/e^2 spot radii in px using the EXISTING Beam_W_Finder.

        The camera's clipped, nonlinear BGR response is not calibrated optical
        irradiance. Treat the numbers as provisional spot-width estimates.
        """
        self.width_result = None
        self.width_crop_bounds = None
        self.width_saturated_pixels = 0
        self.wx_var.set("-- px")
        self.wy_var.set("-- px")

        h, w = signal.shape
        half = WIDTH_HALF_SIZE
        cx, cy = round(xc), round(yc)

        if cx - half < 0 or cy - half < 0 or cx + half > w or cy + half > h:
            self.width_status_var.set("Unavailable: beam near image edge")
            return

        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        crop = signal[y0:y1, x0:x1]

        # The positive-channel difference has a residual noise pedestal.
        # Estimate a constant local pedestal using the crop's outer 5px border.
        border = np.concatenate((
            crop[:5, :].ravel(), crop[-5:, :].ravel(),
            crop[5:-5, :5].ravel(), crop[5:-5, -5:].ravel(),
        ))
        local_background = float(np.median(border))

        try:
            result = measure_beam_width(
                I=crop,
                x=np.arange(x0, x1, dtype=float),
                y=np.arange(y0, y1, dtype=float),
                xc=xc,
                yc=yc,
                background=local_background,
            )
            wx, wy = float(result["wx"]), float(result["wy"])
            if not np.isfinite([wx, wy]).all() or min(wx, wy) <= 0:
                raise ValueError("Invalid radius")

            self.width_result = result
            self.width_crop_bounds = (y0, y1, x0, x1)
            self.wx_var.set(f"{wx:.2f} px")
            self.wy_var.set(f"{wy:.2f} px")

            saturated = int(np.count_nonzero(np.any(
                frame[y0:y1, x0:x1] == 255, axis=2
            )))
            self.width_saturated_pixels = saturated
            warnings = []
            if saturated:
                warnings.append(f"Saturation: {saturated} px")
            if min(wx, wy) < MIN_RELIABLE_RADIUS_PX:
                warnings.append("Spot may be undersampled")
            # The 1/e^2 crossings should not be at the crop boundaries.
            margin = 3.0
            xp, yp = result["x_profile"], result["y_profile"]
            if (xp["left"] < x0 + margin or xp["right"] > x1 - 1 - margin
                    or yp["left"] < y0 + margin or yp["right"] > y1 - 1 - margin):
                warnings.append("Crossing near crop boundary")

            prefix = "Provisional: camera values are not radiometric"
            self.width_status_var.set(
                prefix + (" | " + ", ".join(warnings) if warnings else "")
            )
        except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as exc:
            self.width_status_var.set(f"Width unavailable: {str(exc)[:70]}")

    def analysis_interval(self):
        value = self.rate_var.get()
        if value == "Maximum":
            return 0.0
        return 1.0 / float(value.split()[0])

    def analyze(self, frame):
        started = time.perf_counter()
        self.last_analysis_started = started
        signal = self.extract_signal(frame, self.background)
        detection_image = cv2.GaussianBlur(signal, (7, 7), 1.5)
        peak = float(detection_image.max())
        threshold = max(12.0, 6.0 * self.noise_level)

        self.center = None
        self.roi_bounds = None
        self.x_var.set("-- px")
        self.y_var.set("-- px")
        self.saturation_var.set("--")
        self.width_result = None
        self.width_crop_bounds = None
        self.width_saturated_pixels = 0
        self.wx_var.set("-- px")
        self.wy_var.set("-- px")
        self.width_status_var.set("No laser detected")

        if peak < threshold:
            self.status_var.set("No laser detected")
        else:
            try:
                global_row, global_col = np.unravel_index(
                    np.argmax(detection_image), detection_image.shape
                )
                roi = find_recursive_roi(
                    signal.astype(np.float64),
                    np.arange(signal.shape[1], dtype=float),
                    np.arange(signal.shape[0], dtype=float),
                    iterations=len(ROI_GRID_SIZES),
                    grid_sizes=ROI_GRID_SIZES,
                    anchor=(int(global_row), int(global_col)),
                )

                # Suppress console-only iteration reports without changing the
                # tested Beam_Center_Finder.py implementation.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = refine_center(
                        I_full=signal,
                        x=np.arange(signal.shape[1], dtype=float),
                        y=np.arange(signal.shape[0], dtype=float),
                        roi_bounds=roi["final_bounds"],
                        background=0.0,
                        noise_std=self.noise_level,
                        tolerance=0.1,
                    )

                xc = float(result["center_x"])
                yc = float(result["center_y"])
                if not np.isfinite([xc, yc]).all():
                    raise ValueError("Nonfinite position")
                self.center = (xc, yc)
                self.roi_bounds = roi["final_bounds"]
                self.x_var.set(f"{xc:.2f} px")
                self.y_var.set(f"{yc:.2f} px")
                self.status_var.set("LASER DETECTED")

                r0, r1, c0, c1 = self.roi_bounds
                saturated = int(np.count_nonzero(
                    np.any(frame[r0:r1, c0:c1] == 255, axis=2)
                ))
                self.saturation_var.set(str(saturated))

                # Stage 6 (beta): width in pixels from a local, full-resolution crop.
                # A failure here must never interrupt tracking or the stability test.
                self.measure_width_for_frame(frame, signal, xc, yc)

                if self.stability_active:
                    self.stability_positions.append((xc, yc))
                    self.stability_var.set(
                        f"Collecting: {len(self.stability_positions)}/{STABILITY_SAMPLES}"
                    )
                    if len(self.stability_positions) >= STABILITY_SAMPLES:
                        self.finish_stability()

            except Exception as exc:
                self.status_var.set(f"Analysis failed: {str(exc)[:45]}")
                self.width_status_var.set("Width unavailable: tracking failed")

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.analysis_ms_var.set(f"{elapsed_ms:.1f} ms")
        self.analysis_timestamps.append(started)
        if len(self.analysis_timestamps) >= 2:
            span = self.analysis_timestamps[-1] - self.analysis_timestamps[0]
            if span > 0:
                rate = (len(self.analysis_timestamps) - 1) / span
                self.fps_var.set(f"{rate:.1f} Hz")

        # Heatmap visualization. Contrast scaling is for display ONLY.
        scale = max(20.0, float(np.percentile(signal, 99.95)))
        img_u8 = np.uint8(np.clip(signal * (255.0 / scale), 0, 255))
        self.heatmap = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        self._profile_dirty = True
        self._update_spatial_values()

    def refresh_profile_charts(self):
        """Fit and redraw 1D profiles at up to 5 Hz, not every camera frame.

        A bad fit is nonfatal. The original 1/e^2 width and tracking continue.
        R²/mismatch are diagnostics, not guarantees of optical accuracy.
        """
        if not hasattr(self, "x_profile_plot"):
            return

        self.gaussian_fits = {}
        if self.width_result is None:
            self.x_profile_plot.set_profile(None)
            self.y_profile_plot.set_profile(None)
            self.gwx_var.set("-- px")
            self.gwy_var.set("-- px")
            self.fit_status_var.set("No valid width profile")
        else:
            notes = []
            for axis_name, profile_key, width_key, widget, value_var in (
                ("X", "x_profile", "wx", self.x_profile_plot, self.gwx_var),
                ("Y", "y_profile", "wy", self.y_profile_plot, self.gwy_var),
            ):
                data = self.width_result[profile_key]
                measured_width = float(self.width_result[width_key])
                try:
                    fit = fit_gaussian_profile(
                        data["axis"], data["profile"],
                        radius_guess=measured_width,
                    )
                    self.gaussian_fits[axis_name.lower()] = fit
                    value_var.set(f"{fit['radius']:.2f} px | R² {fit['r_squared']:.3f}")
                    if fit["r_squared"] < FIT_R2_WARNING:
                        notes.append(f"{axis_name}: low R²")
                    difference = fit["relative_width_difference"]
                    if difference is not None and abs(difference) > FIT_WIDTH_MISMATCH_WARNING:
                        notes.append(f"{axis_name}: widths differ >20%")
                except (ValueError, RuntimeError, OverflowError) as exc:
                    fit = None
                    value_var.set("Unavailable")
                    notes.append(f"{axis_name}: fit failed ({str(exc)[:25]})")

                widget.set_profile(data, measured_width, gaussian_fit=fit)

            if self.width_saturated_pixels:
                notes.append(f"{self.width_saturated_pixels} saturated px: unreliable")
            self.fit_status_var.set(
                "Fit checks: " + (" | ".join(notes) if notes else "no flags") +
                "\nR² is not proof of calibrated optical accuracy."
            )

        self._profile_dirty = False
        self._last_profile_refresh = time.perf_counter()
        self._update_spatial_values()

    def finish_stability(self):
        self.stability_active = False
        points = np.asarray(self.stability_positions, dtype=float)
        means = np.mean(points, axis=0)
        stds = np.std(points, axis=0, ddof=1)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"stability_{datetime.now():%Y%m%d_%H%M%S}.csv"
        data = np.column_stack((np.arange(1, len(points) + 1), points))
        np.savetxt(path, data, delimiter=",", header="sample,x_px,y_px",
                   comments="", fmt=["%d", "%.6f", "%.6f"])
        self.stability_var.set(
            f"Mean: ({means[0]:.2f}, {means[1]:.2f}) px\n"
            f"Std: ({stds[0]:.3f}, {stds[1]:.3f}) px\nSaved: {path.name}"
        )

    
    def save_snapshot(self):

        if self.last_frame is None:
            messagebox.showwarning(
                "Snapshot",
                "No camera frame available"
            )
            return

        try:
            # Create output folder if necessary
            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            # Generate filename
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            path = OUTPUT_DIR / f"camera_{timestamp}.png"

            # Encode image as PNG
            success, encoded = cv2.imencode(
                ".png",
                self.last_frame
            )

            if not success:
                raise RuntimeError("PNG encoding failed")

            # Save using Python to support Hebrew paths
            with open(path, "wb") as file:
                file.write(encoded.tobytes())

            # Verify the saved file
            if not path.is_file():
                raise RuntimeError("File was not created")

            self.status_var.set("Snapshot saved")

            messagebox.showinfo(
                "Snapshot Saved",
                f"Image saved successfully:\n\n{path}"
            )

        except Exception as e:

            messagebox.showerror(
                "Save Failed",
                f"Could not save snapshot:\n\n{e}"
            )


    def save_width_profiles(self):
        """Let the user choose where to save the current width profiles."""
        if self.width_result is None:
            messagebox.showinfo(
                "Width profiles",
                "No valid width measurement is available. Turn on the laser, "
                "keep it away from the image edges, and try again.",
            )
            return

        # Save width profiles automatically in the fixed folder.
        folder = WIDTH_OUTPUT_DIR
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not create save folder:\n{exc}")
            self.status_var.set("Width profile save failed")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        try:
            for key, label in (("x_profile", "x"), ("y_profile", "y")):
                data = self.width_result[key]
                path = folder / f"width_{label}_{timestamp}.csv"
                np.savetxt(
                    path,
                    np.column_stack((data["axis"], data["profile"])),
                    delimiter=",",
                    header=f"{label}_px,normalized_signal",
                    comments="",
                    fmt="%.6f",
                )
                saved_paths.append(path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Save failed", f"Could not save width profiles:\n{exc}")
            self.status_var.set("Width profile save failed")
            return

        self.status_var.set(f"Saved width profiles in: {folder}")
        messagebox.showinfo(
            "Saved successfully",
            "The following files were saved:\n\n" +
            "\n".join(str(path) for path in saved_paths),
        )

    @staticmethod
    def show_image(label, image_bgr):
        available_w = label.winfo_width()
        available_h = label.winfo_height()
        if available_w < 20 or available_h < 20:
            return
        h, w = image_bgr.shape[:2]
        scale = min(available_w / w, available_h / h)
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (out_w, out_h), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        label.image = photo  # Keep a strong reference to prevent garbage collection.

    def update(self):
        if not self.running:
            return
        try:
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    self.last_frame = frame
                    if self.calibrating:
                        self.process_calibration_frame(frame)
                    elif self.background is not None and self.tracking_enabled:
                        if time.perf_counter() - self.last_analysis_started >= self.analysis_interval():
                            self.analyze(frame)

                    annotated = frame.copy()
                    if self.roi_bounds is not None:
                        r0, r1, c0, c1 = self.roi_bounds
                        cv2.rectangle(annotated, (c0, r0), (c1 - 1, r1 - 1), (255, 255, 0), 2)
                    if self.center is not None:
                        cv2.drawMarker(
                            annotated,
                            (round(self.center[0]), round(self.center[1])),
                            (0, 255, 0), cv2.MARKER_CROSS, 22, 2,
                        )
                    if self.width_result is not None and self.center is not None:
                        # Draw measured 1/e^2 crossing positions in native pixel coordinates.
                        xp, yp = self.width_result["x_profile"], self.width_result["y_profile"]
                        cx, cy = round(self.center[0]), round(self.center[1])
                        cv2.line(annotated, (round(xp["left"]), cy),
                                 (round(xp["right"]), cy), (0, 210, 255), 2)
                        cv2.line(annotated, (cx, round(yp["left"])),
                                 (cx, round(yp["right"])), (0, 210, 255), 2)
                        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                                       cv2.MARKER_CROSS, 22, 2)
                    self.show_image(self.camera_label, annotated)
                    if self.heatmap is not None:
                        self.show_image(self.heatmap_label, self.heatmap)
                    if (self._profile_dirty and
                            time.perf_counter() - self._last_profile_refresh >= PROFILE_UPDATE_INTERVAL):
                        self.refresh_profile_charts()
                else:
                    self.status_var.set("Camera read failed")
        except Exception as exc:
            self.status_var.set(f"Error: {str(exc)[:45]}")
        finally:
            if self.running:
                self.root.after(15, self.update)

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def main():
    root = tk.Tk()
    LaserProfilerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
