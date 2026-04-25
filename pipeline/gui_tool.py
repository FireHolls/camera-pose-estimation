"""
Camera Pose Estimation — Interface Graphique
Lance :  python pipeline/gui_tool.py
Dep.  :  pip install customtkinter
"""

import sys, os, io, contextlib, threading, copy

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOL_DIR)

try:
    import customtkinter as ctk
except ImportError:
    print("customtkinter missing.  Install with:  pip install customtkinter")
    sys.exit(1)

import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    'figure.facecolor': '#f4f4f4', 'axes.facecolor':  '#ffffff',
    'axes.edgecolor':   '#cccccc', 'axes.labelcolor': '#1e1e2e',
    'xtick.color':      '#1e1e2e', 'ytick.color':     '#1e1e2e',
    'text.color':       '#1e1e2e', 'grid.color':      '#e0e0e0',
    'legend.facecolor': '#f5f5f5', 'legend.edgecolor': '#cccccc',
    'legend.labelcolor': '#1e1e2e',
})

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa

from core import Config, make_scene, run_pipeline, run_noise_sweep, H_RATIO_THRESH
from visualize3d import (
    _draw_3d, _draw_image_cam1,
    _draw_image_cam2_H, _draw_image_cam2_F, _draw_score_bar,
)

# ── Palette Catppuccin Mocha ───────────────────────────────────────────────────
BG      = '#1e1e2e'
SURFACE = '#313244'
OVERLAY = '#45475a'
TEXT    = '#cdd6f4'
SUBTEXT = '#a6adc8'
BLUE    = '#89b4fa'
GREEN   = '#a6e3a1'
RED     = '#f38ba8'
PEACH   = '#fab387'
MAUVE   = '#cba6f7'
TEAL    = '#94e2d5'

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('blue')

FONT_H1  = None
FONT_H2  = None
FONT_NRM = None
FONT_SML = None


# ══════════════════════════════════════════════════════════════════════════════
#  Widgets réutilisables
# ══════════════════════════════════════════════════════════════════════════════

class _SectionTitle(ctk.CTkLabel):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, text=text, font=FONT_H2,
                         text_color=BLUE, anchor='w', **kw)


class EntryRow:
    """Label + CTkEntry sur une ligne de grille."""
    def __init__(self, parent, label, default, row, width=86, colspan=False):
        ctk.CTkLabel(parent, text=label, font=FONT_NRM,
                     anchor='w').grid(row=row, column=0, sticky='w',
                                      padx=(10, 4), pady=3)
        self._var = ctk.StringVar(value=str(default))
        self._entry = ctk.CTkEntry(parent, textvariable=self._var,
                                   width=width, font=FONT_NRM)
        self._entry.grid(row=row, column=1, sticky='e', padx=(4, 10), pady=3)

    def get_float(self, default=0.0):
        try: return float(self._var.get())
        except ValueError: return default

    def get_int(self, default=0):
        try: return int(float(self._var.get()))
        except ValueError: return default

    def set(self, v): self._var.set(str(v))
    def configure(self, **kw): self._entry.configure(**kw)


class TriRow:
    """Label + 3 entrées courtes (ex. rx ry rz)."""
    def __init__(self, parent, label, a, b, c, row, unit=''):
        lbl = f'{label}  ({unit})' if unit else label
        ctk.CTkLabel(parent, text=lbl, font=FONT_NRM,
                     anchor='w').grid(row=row, column=0, sticky='w',
                                      padx=(10, 4), pady=3)
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        frame.grid(row=row, column=1, sticky='e', padx=(4, 10), pady=3)
        self._vars = []
        for i, val in enumerate([a, b, c]):
            v = ctk.StringVar(value=f'{float(val):.4g}')
            ctk.CTkEntry(frame, textvariable=v,
                         width=52, font=FONT_NRM).grid(row=0, column=i, padx=2)
            self._vars.append(v)

    def get_floats(self, defaults=(0., 0., 0.)):
        out = []
        for v, d in zip(self._vars, defaults):
            try: out.append(float(v.get()))
            except ValueError: out.append(d)
        return out

    def set(self, a, b, c):
        for var, val in zip(self._vars, [a, b, c]):
            var.set(f'{float(val):.4g}')


class SliderRow:
    """Label + CTkSlider + CTkEntry liés."""
    def __init__(self, parent, label, default, lo, hi, row,
                 fmt='.2f', steps=200):
        self._fmt = fmt
        ctk.CTkLabel(parent, text=label, font=FONT_NRM,
                     anchor='w').grid(row=row, column=0, sticky='w',
                                      padx=(10, 4), pady=(8, 0))
        f = ctk.CTkFrame(parent, fg_color='transparent')
        f.grid(row=row, column=1, sticky='ew', padx=(4, 10), pady=(8, 0))
        f.columnconfigure(0, weight=1)
        self._sl = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps,
                                  command=self._on_sl)
        self._sl.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self._var = ctk.StringVar()
        self._en = ctk.CTkEntry(f, textvariable=self._var,
                                 width=60, font=FONT_NRM)
        self._en.grid(row=0, column=1)
        self._en.bind('<Return>', self._on_en)
        self._en.bind('<FocusOut>', self._on_en)
        self.set(default)

    def _on_sl(self, v): self._var.set(f'{v:{self._fmt}}')
    def _on_en(self, _=None):
        try: self._sl.set(float(self._var.get()))
        except ValueError: pass

    def get_float(self):
        try: return float(self._var.get())
        except ValueError: return self._sl.get()

    def get_int(self):
        return int(round(self.get_float()))

    def set(self, v):
        self._sl.set(v)
        self._var.set(f'{v:{self._fmt}}')


# ══════════════════════════════════════════════════════════════════════════════
#  Application principale
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  Application principale
# ══════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        global FONT_H1, FONT_H2, FONT_NRM, FONT_SML
        FONT_H1  = ctk.CTkFont(family='Segoe UI', size=14, weight='bold')
        FONT_H2  = ctk.CTkFont(family='Segoe UI', size=11, weight='bold')
        FONT_NRM = ctk.CTkFont(family='Segoe UI', size=11)
        FONT_SML = ctk.CTkFont(family='Segoe UI', size=10)
        self.title('Camera Pose Estimation — Research Tool')
        self.geometry('1560x940')
        self.minsize(1200, 720)
        self.configure(fg_color=BG)
        self._running = False

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_header()
        self._build_sidebar()
        self._build_canvas()
        self._build_footer()

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        bar = ctk.CTkFrame(self, height=52, corner_radius=0,
                           fg_color=SURFACE)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew')
        bar.columnconfigure(1, weight=1)

        ctk.CTkLabel(bar, text='📷  Camera Pose Estimation',
                     font=FONT_H1, text_color=BLUE).pack(side='left', padx=20)
        ctk.CTkLabel(bar, text='Research Tool  ·  H vs F  ·  ORB-SLAM style',
                     font=FONT_SML, text_color=SUBTEXT).pack(side='left')

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = ctk.CTkScrollableFrame(self, width=390, corner_radius=0,
                                    fg_color=SURFACE,
                                    scrollbar_button_color=OVERLAY)
        sb.grid(row=1, column=0, sticky='nsew')
        sb.columnconfigure(0, weight=1)

        tabs = ctk.CTkTabview(sb, width=375, fg_color=BG,
                              segmented_button_selected_color=BLUE,
                              segmented_button_selected_hover_color='#6ca0e8',
                              segmented_button_unselected_color=OVERLAY,
                              text_color=TEXT)
        tabs.pack(fill='both', expand=True, padx=6, pady=8)

        for t in ('Scene', 'Cameras', 'Optics', 'Analysis'):
            tabs.add(t)

        self._tab_scene(tabs.tab('Scene'))
        self._tab_cameras(tabs.tab('Cameras'))
        self._tab_optics(tabs.tab('Optics'))
        self._tab_analyse(tabs.tab('Analysis'))

    # ── Tab Scene ─────────────────────────────────────────────────────────────

    def _tab_scene(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Scene type').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 2)); r += 1

        self._scene_type_var = ctk.StringVar(value='Planar')
        seg = ctk.CTkSegmentedButton(
            tab, values=['Planar', 'Non-planar'],
            variable=self._scene_type_var, font=FONT_NRM,
            selected_color=BLUE, selected_hover_color='#6ca0e8',
            command=self._on_scene_type)
        seg.grid(row=r, column=0, columnspan=2, sticky='ew',
                 padx=10, pady=4); r += 1

        # Planaire : Z fixe
        self._z_frame = ctk.CTkFrame(tab, fg_color='transparent')
        self._z_frame.grid(row=r, column=0, columnspan=2, sticky='ew')
        self._z_frame.columnconfigure(1, weight=1)
        self._z_min = EntryRow(self._z_frame, 'Depth Z  (m)', 5.0, 0)

        # Non-planar: Z min + Z max (hidden by default)
        self._z_range_frame = ctk.CTkFrame(tab, fg_color='transparent')
        self._z_range_frame.grid(row=r, column=0, columnspan=2, sticky='ew')
        self._z_range_frame.columnconfigure(1, weight=1)
        self._z_min_np = EntryRow(self._z_range_frame, 'Z min  (m)', 3.0, 0)
        self._z_max_np = EntryRow(self._z_range_frame, 'Z max  (m)', 7.0, 1)
        self._z_range_frame.grid_remove()
        r += 1

        _SectionTitle(tab, '  Point cloud').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._x_range = EntryRow(tab, 'Range X  ±(m)', 2.0, r); r += 1
        self._y_range = EntryRow(tab, 'Range Y  ±(m)', 1.5, r); r += 1
        self._n_pts   = SliderRow(tab, 'Number of points',
                                   100, 8, 500, r, fmt='.0f', steps=492); r += 2

        _SectionTitle(tab, '  Signal degradation').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._noise = SliderRow(tab, 'Pixel noise  σ (px)  [cam 1 & 2]',
                                 0.0, 0.0, 10.0, r, fmt='.2f'); r += 2
        self._outl  = SliderRow(tab, 'Outliers  (%)  [cam 2 only]',
                                 0.0, 0.0, 50.0, r, fmt='.1f'); r += 2

    def _on_scene_type(self, val):
        if val == 'Planar':
            self._z_frame.grid()
            self._z_range_frame.grid_remove()
        else:
            self._z_frame.grid_remove()
            self._z_range_frame.grid()

    # ── Tab Cameras ───────────────────────────────────────────────────────────

    def _tab_cameras(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Camera 1  —  Reference').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 4)); r += 1

        self._c1_rot   = TriRow(tab, 'Rotation   rx ry rz', 0, 0, 0, r, 'deg'); r += 1
        self._c1_trans = TriRow(tab, 'Translation tx ty tz', 0, 0, 0, r, 'm');   r += 1

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew',
            padx=10, pady=10); r += 1

        _SectionTitle(tab, '  Camera 2').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 4)); r += 1

        self._c2_rot   = TriRow(tab, 'Rotation   rx ry rz', 0, 8, 0, r, 'deg'); r += 1
        self._c2_trans = TriRow(tab, 'Translation tx ty tz', 0.4, 0, 0, r, 'm'); r += 1

        info = ctk.CTkTextbox(tab, height=70, font=FONT_SML,
                               fg_color=OVERLAY, text_color=SUBTEXT,
                               border_width=0, state='normal')
        info.grid(row=r, column=0, columnspan=2, sticky='ew',
                  padx=10, pady=(12, 4)); r += 1
        info.insert('1.0',
            'Frame: X_cam = R @ X_world + t\n'
            'R = Rz·Ry·Rx  (extrinsic Euler, deg)\n'
            'Cam 1 identity by default → world reference')
        info.configure(state='disabled')

    # ── Tab Optiques ──────────────────────────────────────────────────────────

    def _tab_optics(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Focal lengths').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 2)); r += 1

        self._fx = EntryRow(tab, 'fx  (px)', 1000, r); r += 1
        self._fy = EntryRow(tab, 'fy  (px)', 1000, r); r += 1

        _SectionTitle(tab, '  Principal point').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._cx = EntryRow(tab, 'cx  (px)', 960, r); r += 1
        self._cy = EntryRow(tab, 'cy  (px)', 540, r); r += 1

        _SectionTitle(tab, '  Image resolution').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._img_w = EntryRow(tab, 'Width  W  (px)', 1920, r); r += 1
        self._img_h = EntryRow(tab, 'Height  H  (px)', 1080, r); r += 1

        # FOV live display
        self._fov_label = ctk.CTkLabel(tab, text='', font=FONT_SML,
                                        text_color=TEAL)
        self._fov_label.grid(row=r, column=0, columnspan=2, sticky='w',
                              padx=10, pady=(10, 4)); r += 1
        self._update_fov()

        for entry in [self._fx, self._fy, self._cx, self._cy,
                      self._img_w, self._img_h]:
            entry._entry.bind('<FocusOut>', lambda _: self._update_fov())
            entry._entry.bind('<Return>',   lambda _: self._update_fov())

    def _update_fov(self, _=None):
        try:
            fx = self._fx.get_float(1000)
            fy = self._fy.get_float(1000)
            w  = self._img_w.get_int(1920)
            h  = self._img_h.get_int(1080)
            fh = 2 * np.degrees(np.arctan(w / (2 * fx)))
            fv = 2 * np.degrees(np.arctan(h / (2 * fy)))
            self._fov_label.configure(text=f'FOV  →  {fh:.1f}° × {fv:.1f}°')
        except Exception:
            pass

    # ── Tab Analyse ───────────────────────────────────────────────────────────

    def _tab_analyse(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Methods').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 4)); r += 1

        self._use_h = ctk.CTkSwitch(tab, text='Homography H',
                                     font=FONT_NRM, onvalue=1, offvalue=0,
                                     button_color=BLUE, progress_color=BLUE)
        self._use_h.select()
        self._use_h.grid(row=r, column=0, columnspan=2, sticky='w',
                          padx=10, pady=3); r += 1

        self._use_f = ctk.CTkSwitch(tab, text='Fundamental matrix F  (8 pts)',
                                     font=FONT_NRM, onvalue=1, offvalue=0,
                                     button_color=BLUE, progress_color=BLUE)
        self._use_f.select()
        self._use_f.grid(row=r, column=0, columnspan=2, sticky='w',
                          padx=10, pady=3); r += 1

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew', padx=10, pady=10); r += 1

        _SectionTitle(tab, '  Analysis mode').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 4)); r += 1

        self._mode_var = ctk.StringVar(value='Single scene')
        mode_seg = ctk.CTkSegmentedButton(
            tab, values=['Single scene', 'Noise sweep'],
            variable=self._mode_var, font=FONT_NRM,
            selected_color=MAUVE, selected_hover_color='#b09bdf',
            command=self._on_mode)
        mode_seg.grid(row=r, column=0, columnspan=2, sticky='ew',
                      padx=10, pady=4); r += 1

        self._noise_frame = ctk.CTkFrame(tab, fg_color='transparent')
        self._noise_frame.grid(row=r, column=0, columnspan=2, sticky='ew'); r += 1
        self._noise_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(self._noise_frame, text='σ levels  (space-separated)',
                     font=FONT_NRM, anchor='w').grid(
            row=0, column=0, sticky='w', padx=(10, 4), pady=3)
        self._noise_levels_var = ctk.StringVar(value='0 0.5 1 2 3 5')
        ctk.CTkEntry(self._noise_frame, textvariable=self._noise_levels_var,
                     font=FONT_NRM).grid(row=0, column=1, sticky='ew',
                                          padx=(4, 10), pady=3)

        self._noise_frame.grid_remove()

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew', padx=10, pady=10); r += 1

        self._export = ctk.CTkSwitch(tab, text='Save figure (PNG)',
                                      font=FONT_NRM, onvalue=1, offvalue=0,
                                      button_color=GREEN, progress_color=GREEN)
        self._export.grid(row=r, column=0, columnspan=2, sticky='w',
                           padx=10, pady=3); r += 1

        # Results card (updated after run)
        self._res_box = ctk.CTkTextbox(tab, height=120, font=FONT_SML,
                                        fg_color=OVERLAY, text_color=TEXT,
                                        border_width=0, state='disabled')
        self._res_box.grid(row=r, column=0, columnspan=2, sticky='ew',
                            padx=10, pady=(14, 6))

    def _on_mode(self, val):
        self._noise_frame.grid_remove()
        if val == 'Noise sweep':
            self._noise_frame.grid()

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        CANVAS_BG = '#f4f4f4'
        TOOLBAR_BG = '#e8e8e8'

        self._cv_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=CANVAS_BG)
        self._cv_frame.grid(row=1, column=1, sticky='nsew')
        self._cv_frame.rowconfigure(1, weight=1)
        self._cv_frame.columnconfigure(0, weight=1)

        # Toolbar row
        self._tb_frame = ctk.CTkFrame(self._cv_frame, height=32,
                                       corner_radius=0, fg_color=TOOLBAR_BG)
        self._tb_frame.grid(row=0, column=0, sticky='ew')

        # Matplotlib figure
        self._fig = Figure(facecolor=CANVAS_BG)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self._cv_frame)
        self._mpl_canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self._toolbar = NavigationToolbar2Tk(
            self._mpl_canvas, self._tb_frame, pack_toolbar=False)
        try:
            self._toolbar.config(background=TOOLBAR_BG)
            for w in self._toolbar.winfo_children():
                try: w.config(background=TOOLBAR_BG, foreground='#1e1e2e')
                except Exception: pass
        except Exception:
            pass
        self._toolbar.update()
        self._toolbar.pack(side='left', padx=4)

        self._show_placeholder()

    def _show_placeholder(self):
        self._fig.clear()
        self._fig.set_facecolor('#f4f4f4')
        ax = self._fig.add_subplot(111)
        ax.set_facecolor('#f4f4f4')
        ax.text(0.5, 0.52,
                'Configure parameters\nand click  Run',
                ha='center', va='center', fontsize=17,
                color='#aaaaaa', transform=ax.transAxes,
                fontfamily='Segoe UI')
        ax.axis('off')
        self._mpl_canvas.draw_idle()

    # ── Footer ────────────────────────────────────────────────────────────────

    def _build_footer(self):
        bar = ctk.CTkFrame(self, height=48, corner_radius=0,
                           fg_color='#11111b')
        bar.grid(row=2, column=0, columnspan=2, sticky='ew')
        bar.columnconfigure(2, weight=1)

        self._run_btn = ctk.CTkButton(
            bar, text='  Run', width=130, height=34,
            font=ctk.CTkFont(family='Segoe UI', size=13, weight='bold'),
            fg_color=BLUE, hover_color='#6ca0e8',
            command=self.on_run)
        self._run_btn.pack(side='left', padx=(14, 6), pady=7)

        ctk.CTkButton(
            bar, text='Reset', width=120, height=34,
            font=FONT_NRM, fg_color=OVERLAY, hover_color='#585b70',
            command=self._reset).pack(side='left', padx=4, pady=7)

        self._prog = ctk.CTkProgressBar(bar, width=150, height=10,
                                         progress_color=TEAL)
        self._prog.set(0)
        self._prog.pack(side='left', padx=14, pady=7)
        self._prog.pack_forget()

        self._status = ctk.CTkLabel(
            bar, text='Ready.', font=FONT_SML,
            text_color=SUBTEXT, anchor='w')
        self._status.pack(side='left', padx=8, fill='x', expand=True)

    # ── Config collect ────────────────────────────────────────────────────────

    def _collect_config(self):
        cfg = Config()

        st_map = {'Planar': 'planar', 'Non-planar': 'nonplanar'}
        cfg.scene_type = st_map[self._scene_type_var.get()]
        if cfg.scene_type == 'planar':
            cfg.z_min = self._z_min.get_float(5.0)
        else:
            cfg.z_min = self._z_min_np.get_float(3.0)
            cfg.z_max = self._z_max_np.get_float(7.0)
        cfg.x_range       = self._x_range.get_float(2.0)
        cfg.y_range       = self._y_range.get_float(1.5)
        cfg.n_points      = self._n_pts.get_int()
        cfg.noise_sigma   = self._noise.get_float()
        cfg.outlier_ratio = self._outl.get_float() / 100.0
        cfg.seed          = 42

        cfg.cam1_rx, cfg.cam1_ry, cfg.cam1_rz = self._c1_rot.get_floats()
        cfg.cam1_tx, cfg.cam1_ty, cfg.cam1_tz = self._c1_trans.get_floats()
        cfg.cam2_rx, cfg.cam2_ry, cfg.cam2_rz = self._c2_rot.get_floats()
        cfg.cam2_tx, cfg.cam2_ty, cfg.cam2_tz = self._c2_trans.get_floats()

        cfg.fx    = self._fx.get_float(1000.)
        cfg.fy    = self._fy.get_float(1000.)
        cfg.cx    = self._cx.get_float(960.)
        cfg.cy    = self._cy.get_float(540.)
        cfg.img_w = self._img_w.get_int(1920)
        cfg.img_h = self._img_h.get_int(1080)

        cfg.use_H = self._use_h.get() == 1
        cfg.use_F = self._use_f.get() == 1

        mode_map = {'Single scene': 'single', 'Noise sweep': 'noise_sweep'}
        cfg.mode = mode_map[self._mode_var.get()]

        try:
            cfg.noise_levels = sorted(
                [float(x) for x in self._noise_levels_var.get().split()])
        except ValueError:
            pass

        cfg.export_png = self._export.get() == 1
        return cfg

    # ── Lancer l'analyse ──────────────────────────────────────────────────────

    def on_run(self):
        if self._running:
            return
        cfg = self._collect_config()
        if not cfg.use_H and not cfg.use_F:
            self._set_status('Enable at least one method (H or F).', RED)
            return

        self._running = True
        self._run_btn.configure(state='disabled', text='  Running…')
        self._prog.set(0)
        self._prog.pack(side='left', padx=14, pady=7)
        self._set_status('Analysis running…', PEACH)

        def thread():
            try:
                if cfg.mode == 'single':
                    scene = make_scene(cfg)
                    res   = run_pipeline(scene, cfg)
                    self.after(0, lambda: self._show_single(scene, res, cfg))

                elif cfg.mode == 'noise_sweep':
                    data = run_noise_sweep(cfg)
                    self.after(0, lambda: self._show_noise(data, cfg))

            except Exception as exc:
                self.after(0, lambda e=exc:
                           self._set_status(f'Error: {e}', RED))
            finally:
                self.after(0, self._done)

        threading.Thread(target=thread, daemon=True).start()

    def _done(self):
        self._running = False
        self._run_btn.configure(state='normal', text='  Run')
        self._prog.pack_forget()

    def _set_status(self, msg, color=SUBTEXT):
        self._status.configure(text=msg, text_color=color)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _style_ax2d(self, ax):
        ax.set_facecolor('#ffffff')

    def _polish_ax2d(self, ax):
        pass  # rcParams clairs gèrent tout

    def _style_ax3d(self, ax):
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = True
            pane.set_facecolor('#efefef')
            pane.set_edgecolor('#cccccc')
        ax.grid(True, color='#dddddd', alpha=1.0)

    def _polish_ax3d(self, ax):
        pass  # rcParams clairs gèrent tout

    def _show_single(self, scene, res, cfg):
        self._fig.clear()
        self._fig.set_facecolor('#f4f4f4')
        gs = self._fig.add_gridspec(
            4, 2, width_ratios=[1.7, 1.0],
            height_ratios=[1, 1, 1, 1],
            hspace=0.38, wspace=0.18,
        )
        self._fig.subplots_adjust(
            top=0.95, bottom=0.04, left=0.04, right=0.99)

        ax3d   = self._fig.add_subplot(gs[:, 0], projection='3d')
        ax_c1  = self._fig.add_subplot(gs[0, 1])
        ax_c2h = self._fig.add_subplot(gs[1, 1])
        ax_c2f = self._fig.add_subplot(gs[2, 1])
        ax_bar = self._fig.add_subplot(gs[3, 1])

        self._style_ax3d(ax3d)
        for ax in [ax_c1, ax_c2h, ax_c2f, ax_bar]:
            self._style_ax2d(ax)

        w, h = cfg.img_w, cfg.img_h
        win  = res.get('winner') or 'F'

        _draw_3d(ax3d, scene, res['R_H'], res['t_H'],
                 res['R_F'], res['t_F'], win,
                 cfg.scene_type == 'planar', w=w, h=h)
        _draw_image_cam1(ax_c1, scene['px1'], w=w, h=h)
        _draw_image_cam2_H(ax_c2h, scene['px2'], scene['pts3d'],
                            res['R_H'], res['t_H'], scene['K'], win, w=w, h=h)
        _draw_image_cam2_F(ax_c2f, scene['px2'], scene['pts3d'],
                            res['R_F'], res['t_F'], scene['K'], win, w=w, h=h)
        _draw_score_bar(ax_bar, res['S_H'] or 0, res['S_F'] or 0,
                        win, res['ratio'] or 0, H_RATIO_THRESH)

        self._polish_ax3d(ax3d)
        for ax in [ax_c1, ax_c2h, ax_c2f, ax_bar]:
            self._polish_ax2d(ax)

        scene_lbl = 'Planar' if cfg.scene_type == 'planar' else 'Non-planar'
        self._fig.suptitle(
            f'Scene {scene_lbl}  ·  {res["n_visible"]} visible pts  ·  '
            f'Winner [{win}]',
            fontsize=12, fontweight='bold', color='#1e1e2e', y=0.995)

        self._mpl_canvas.draw_idle()
        self._export_if_needed(cfg, 'single')

        lines = [f'Visible points: {res["n_visible"]}']
        if res['S_H'] is not None:
            lines.append(
                f'H  →  Err R = {res["err_R_H"]:.3f}°  '
                f'|  Err t = {res["err_t_H"]:.3f}°')
        if res['S_F'] is not None:
            lines.append(
                f'F  →  Err R = {res["err_R_F"]:.3f}°  '
                f'|  Err t = {res["err_t_F"]:.3f}°')
        if res['ratio'] is not None:
            lines.append(f'Ratio = {res["ratio"]:.3f}  →  [{win}]')
        self._update_res_box('\n'.join(lines))
        self._set_status(
            f'Analysis done  ·  {res["n_visible"]} pts  ·  [{win}]', GREEN)

    def _show_noise(self, data, cfg):
        self._fig.clear()
        self._fig.set_facecolor('#f4f4f4')
        fig = self._fig

        gs = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.25)
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.98)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self._style_ax2d(ax1)
        self._style_ax2d(ax2)

        levels = [d['sigma'] for d in data]
        H_C, F_C = '#fab387', '#f38ba8'

        for ax, key_H, key_F, ylabel, title in [
            (ax1, 'err_R_H', 'err_R_F',
             'Rotation error (°)', 'Rotation'),
            (ax2, 'err_t_H', 'err_t_F',
             'Translation direction error (°)', 'Translation'),
        ]:
            if cfg.use_H:
                ax.plot(levels, [d[key_H] for d in data],
                        'o-', color=H_C, lw=2, ms=6, label='H')
            if cfg.use_F:
                ax.plot(levels, [d[key_F] for d in data],
                        's-', color=F_C, lw=2, ms=6, label='F')
            ax.set_xlabel('σ  (px)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()

        for ax in [ax1, ax2]:
            self._polish_ax2d(ax)

        fig.suptitle('Noise robustness', fontsize=13,
                     fontweight='bold', color='#1e1e2e')
        self._mpl_canvas.draw_idle()
        self._export_if_needed(cfg, 'noise_sweep')
        self._update_res_box(
            f'Sweep: {len(levels)} levels\n'
            f'σ from {min(levels):.1f} to {max(levels):.1f} px')
        self._set_status('Noise sweep done.', GREEN)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_res_box(self, text):
        self._res_box.configure(state='normal')
        self._res_box.delete('1.0', 'end')
        self._res_box.insert('1.0', text)
        self._res_box.configure(state='disabled')

    def _export_if_needed(self, cfg, suffix):
        if not cfg.export_png:
            return
        from datetime import datetime
        export_dir = os.path.join(TOOL_DIR, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = os.path.join(export_dir, f'result_{suffix}_{ts}.png')
        self._fig.savefig(out, dpi=150, bbox_inches='tight',
                          facecolor=self._fig.get_facecolor())
        self._set_status(f'Figure saved: exports/{os.path.basename(out)}', TEAL)

    def _reset(self):
        defaults = Config()
        self._scene_type_var.set('Planar')
        self._z_min.set(defaults.z_min)
        self._z_min_np.set(3.0)
        self._z_max_np.set(7.0)
        self._on_scene_type('Planar')
        self._x_range.set(defaults.x_range)
        self._y_range.set(defaults.y_range)
        self._n_pts.set(defaults.n_points)
        self._noise.set(defaults.noise_sigma)
        self._outl.set(defaults.outlier_ratio * 100)
        self._c1_rot.set(defaults.cam1_rx, defaults.cam1_ry, defaults.cam1_rz)
        self._c1_trans.set(defaults.cam1_tx, defaults.cam1_ty, defaults.cam1_tz)
        self._c2_rot.set(defaults.cam2_rx, defaults.cam2_ry, defaults.cam2_rz)
        self._c2_trans.set(defaults.cam2_tx, defaults.cam2_ty, defaults.cam2_tz)
        self._fx.set(defaults.fx); self._fy.set(defaults.fy)
        self._cx.set(defaults.cx); self._cy.set(defaults.cy)
        self._img_w.set(defaults.img_w); self._img_h.set(defaults.img_h)
        self._use_h.select(); self._use_f.select()
        self._noise_levels_var.set('0 0.5 1 2 3 5')
        self._mode_var.set('Single scene')
        self._on_mode('Single scene')
        self._update_fov()
        self._show_placeholder()
        self._set_status('Parameters reset.', SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
