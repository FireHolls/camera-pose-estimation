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
    print("customtkinter manquant.  Installez avec :  pip install customtkinter")
    sys.exit(1)

import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    'figure.facecolor': '#1e1e2e', 'axes.facecolor':  '#181825',
    'axes.edgecolor':   '#45475a', 'axes.labelcolor': '#cdd6f4',
    'xtick.color':      '#cdd6f4', 'ytick.color':     '#cdd6f4',
    'text.color':       '#cdd6f4', 'grid.color':      '#313244',
    'legend.facecolor': '#1e1e2e', 'legend.edgecolor': '#45475a',
    'legend.labelcolor': '#cdd6f4',
})

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa

from research_tool import Config, make_scene, run_pipeline, H_RATIO_THRESH
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
#  Fonctions de calcul (exécutées dans un thread)
# ══════════════════════════════════════════════════════════════════════════════

def _noise_sweep(cfg):
    levels, results = sorted(cfg.noise_levels), []
    tmp = copy.copy(cfg)
    for sigma in levels:
        tmp.noise_sigma = sigma
        scene = make_scene(tmp)
        res   = run_pipeline(scene, tmp)
        results.append({'sigma': sigma, **res})
    return results


def _montecarlo(cfg, progress_cb=None):
    results = []
    for i in range(cfg.mc_n):
        scene = make_scene(cfg, seed_override=cfg.seed + i)
        res   = run_pipeline(scene, cfg)
        results.append(res)
        if progress_cb:
            progress_cb(i + 1, cfg.mc_n)
    return results


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

        for t in ('Scene', 'Cameras', 'Optiques', 'Analyse'):
            tabs.add(t)

        self._tab_scene(tabs.tab('Scene'))
        self._tab_cameras(tabs.tab('Cameras'))
        self._tab_optics(tabs.tab('Optiques'))
        self._tab_analyse(tabs.tab('Analyse'))

    # ── Tab Scene ─────────────────────────────────────────────────────────────

    def _tab_scene(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Type de scène').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 2)); r += 1

        self._scene_type_var = ctk.StringVar(value='Planaire')
        seg = ctk.CTkSegmentedButton(
            tab, values=['Planaire', 'Non-planaire', 'Custom'],
            variable=self._scene_type_var, font=FONT_NRM,
            selected_color=BLUE, selected_hover_color='#6ca0e8',
            command=self._on_scene_type)
        seg.grid(row=r, column=0, columnspan=2, sticky='ew',
                 padx=10, pady=4); r += 1

        self._z_min = EntryRow(tab, 'Profondeur Z  (m)', 5.0, r); r += 1
        self._z_max = EntryRow(tab, 'Z max  (m)', 7.0, r); r += 1
        self._z_max.configure(state='disabled')

        _SectionTitle(tab, '  Nuage de points').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._x_range = EntryRow(tab, 'Étendue X  ±(m)', 2.0, r); r += 1
        self._y_range = EntryRow(tab, 'Étendue Y  ±(m)', 1.5, r); r += 1
        self._n_pts   = SliderRow(tab, 'Nombre de points',
                                   100, 8, 500, r, fmt='.0f', steps=492); r += 2

        _SectionTitle(tab, '  Dégradation du signal').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._noise = SliderRow(tab, 'Bruit pixel  σ (px)',
                                 0.0, 0.0, 10.0, r, fmt='.2f'); r += 2
        self._outl  = SliderRow(tab, 'Outliers  (%)',
                                 0.0, 0.0, 50.0, r, fmt='.1f'); r += 2
        self._seed  = EntryRow(tab, 'Seed', 42, r); r += 1

    def _on_scene_type(self, val):
        if val == 'Custom':
            self._z_max.configure(state='normal')
        else:
            self._z_max.configure(state='disabled')

    # ── Tab Cameras ───────────────────────────────────────────────────────────

    def _tab_cameras(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Caméra 1  —  Référence').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 4)); r += 1

        self._c1_rot   = TriRow(tab, 'Rotation   rx ry rz', 0, 0, 0, r, 'deg'); r += 1
        self._c1_trans = TriRow(tab, 'Translation tx ty tz', 0, 0, 0, r, 'm');   r += 1

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew',
            padx=10, pady=10); r += 1

        _SectionTitle(tab, '  Caméra 2').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 4)); r += 1

        self._c2_rot   = TriRow(tab, 'Rotation   rx ry rz', 0, 8, 0, r, 'deg'); r += 1
        self._c2_trans = TriRow(tab, 'Translation tx ty tz', 0.4, 0, 0, r, 'm'); r += 1

        # Info box
        info = ctk.CTkTextbox(tab, height=70, font=FONT_SML,
                               fg_color=OVERLAY, text_color=SUBTEXT,
                               border_width=0, state='normal')
        info.grid(row=r, column=0, columnspan=2, sticky='ew',
                  padx=10, pady=(12, 4)); r += 1
        info.insert('1.0',
            'Repère : X_cam = R @ X_world + t\n'
            'R = Rz·Ry·Rx  (Euler extrinsèque, deg)\n'
            'Cam 1 identité par défaut → référence mondiale')
        info.configure(state='disabled')

    # ── Tab Optiques ──────────────────────────────────────────────────────────

    def _tab_optics(self, tab):
        tab.columnconfigure(1, weight=1)
        r = 0

        _SectionTitle(tab, '  Focales').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 2)); r += 1

        self._fx = EntryRow(tab, 'fx  (px)', 1000, r); r += 1
        self._fy = EntryRow(tab, 'fy  (px)', 1000, r); r += 1

        _SectionTitle(tab, '  Point principal').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._cx = EntryRow(tab, 'cx  (px)', 960, r); r += 1
        self._cy = EntryRow(tab, 'cy  (px)', 540, r); r += 1

        _SectionTitle(tab, '  Résolution image').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2)); r += 1

        self._img_w = EntryRow(tab, 'Largeur  W  (px)', 1920, r); r += 1
        self._img_h = EntryRow(tab, 'Hauteur  H  (px)', 1080, r); r += 1

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

        _SectionTitle(tab, '  Méthodes').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(10, 4)); r += 1

        self._use_h = ctk.CTkSwitch(tab, text='Homographie H',
                                     font=FONT_NRM, onvalue=1, offvalue=0,
                                     button_color=BLUE, progress_color=BLUE)
        self._use_h.select()
        self._use_h.grid(row=r, column=0, columnspan=2, sticky='w',
                          padx=10, pady=3); r += 1

        self._use_f = ctk.CTkSwitch(tab, text='Matrice fondamentale F  (8 pts)',
                                     font=FONT_NRM, onvalue=1, offvalue=0,
                                     button_color=BLUE, progress_color=BLUE)
        self._use_f.select()
        self._use_f.grid(row=r, column=0, columnspan=2, sticky='w',
                          padx=10, pady=3); r += 1

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew', padx=10, pady=10); r += 1

        _SectionTitle(tab, "  Mode d'analyse").grid(
            row=r, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 4)); r += 1

        self._mode_var = ctk.StringVar(value='Scene unique')
        mode_seg = ctk.CTkSegmentedButton(
            tab, values=['Scene unique', 'Balayage bruit', 'Monte-Carlo'],
            variable=self._mode_var, font=FONT_NRM,
            selected_color=MAUVE, selected_hover_color='#b09bdf',
            command=self._on_mode)
        mode_seg.grid(row=r, column=0, columnspan=2, sticky='ew',
                      padx=10, pady=4); r += 1

        # Noise sweep params
        self._noise_frame = ctk.CTkFrame(tab, fg_color='transparent')
        self._noise_frame.grid(row=r, column=0, columnspan=2, sticky='ew'); r += 1
        self._noise_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(self._noise_frame, text='Niveaux σ  (espaces)',
                     font=FONT_NRM, anchor='w').grid(
            row=0, column=0, sticky='w', padx=(10, 4), pady=3)
        self._noise_levels_var = ctk.StringVar(value='0 0.5 1 2 3 5')
        ctk.CTkEntry(self._noise_frame, textvariable=self._noise_levels_var,
                     font=FONT_NRM).grid(row=0, column=1, sticky='ew',
                                          padx=(4, 10), pady=3)

        # Monte-Carlo params
        self._mc_frame = ctk.CTkFrame(tab, fg_color='transparent')
        self._mc_frame.grid(row=r, column=0, columnspan=2, sticky='ew'); r += 1
        self._mc_frame.columnconfigure(1, weight=1)
        self._mc_n = SliderRow(self._mc_frame, 'Itérations N',
                                50, 10, 500, 0, fmt='.0f', steps=490)

        self._noise_frame.grid_remove()
        self._mc_frame.grid_remove()

        ctk.CTkFrame(tab, height=1, fg_color=OVERLAY).grid(
            row=r, column=0, columnspan=2, sticky='ew', padx=10, pady=10); r += 1

        self._export = ctk.CTkSwitch(tab, text='Sauvegarder figure (PNG)',
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
        self._mc_frame.grid_remove()
        if val == 'Balayage bruit':
            self._noise_frame.grid()
        elif val == 'Monte-Carlo':
            self._mc_frame.grid()

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        self._cv_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=BG)
        self._cv_frame.grid(row=1, column=1, sticky='nsew')
        self._cv_frame.rowconfigure(1, weight=1)
        self._cv_frame.columnconfigure(0, weight=1)

        # Toolbar row
        self._tb_frame = ctk.CTkFrame(self._cv_frame, height=32,
                                       corner_radius=0, fg_color='#11111b')
        self._tb_frame.grid(row=0, column=0, sticky='ew')

        # Matplotlib figure
        self._fig = Figure(facecolor='#1e1e2e')
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self._cv_frame)
        self._mpl_canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self._toolbar = NavigationToolbar2Tk(
            self._mpl_canvas, self._tb_frame, pack_toolbar=False)
        try:
            self._toolbar.config(background='#11111b')
            for w in self._toolbar.winfo_children():
                try: w.config(background='#11111b', foreground=TEXT)
                except Exception: pass
        except Exception:
            pass
        self._toolbar.update()
        self._toolbar.pack(side='left', padx=4)

        self._show_placeholder()

    def _show_placeholder(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor('#181825')
        ax.text(0.5, 0.52,
                'Configurez les paramètres\net cliquez sur  Analyser',
                ha='center', va='center', fontsize=17,
                color='#45475a', transform=ax.transAxes,
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
            bar, text='  Analyser', width=130, height=34,
            font=ctk.CTkFont(family='Segoe UI', size=13, weight='bold'),
            fg_color=BLUE, hover_color='#6ca0e8',
            command=self.on_run)
        self._run_btn.pack(side='left', padx=(14, 6), pady=7)

        ctk.CTkButton(
            bar, text='Réinitialiser', width=120, height=34,
            font=FONT_NRM, fg_color=OVERLAY, hover_color='#585b70',
            command=self._reset).pack(side='left', padx=4, pady=7)

        self._prog = ctk.CTkProgressBar(bar, width=150, height=10,
                                         progress_color=TEAL)
        self._prog.set(0)
        self._prog.pack(side='left', padx=14, pady=7)
        self._prog.pack_forget()

        self._status = ctk.CTkLabel(
            bar, text='Prêt.', font=FONT_SML,
            text_color=SUBTEXT, anchor='w')
        self._status.pack(side='left', padx=8, fill='x', expand=True)

    # ── Config collect ────────────────────────────────────────────────────────

    def _collect_config(self):
        cfg = Config()

        st_map = {'Planaire': 'planar', 'Non-planaire': 'nonplanar', 'Custom': 'custom'}
        cfg.scene_type    = st_map[self._scene_type_var.get()]
        cfg.z_min         = self._z_min.get_float(5.0)
        cfg.z_max         = self._z_max.get_float(7.0)
        cfg.x_range       = self._x_range.get_float(2.0)
        cfg.y_range       = self._y_range.get_float(1.5)
        cfg.n_points      = self._n_pts.get_int()
        cfg.noise_sigma   = self._noise.get_float()
        cfg.outlier_ratio = self._outl.get_float() / 100.0
        cfg.seed          = self._seed.get_int(42)

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

        mode_map = {'Scene unique': 'single',
                    'Balayage bruit': 'noise_sweep',
                    'Monte-Carlo': 'montecarlo'}
        cfg.mode = mode_map[self._mode_var.get()]

        try:
            cfg.noise_levels = sorted(
                [float(x) for x in self._noise_levels_var.get().split()])
        except ValueError:
            pass

        cfg.mc_n       = self._mc_n.get_int()
        cfg.export_png = self._export.get() == 1
        return cfg

    # ── Lancer l'analyse ──────────────────────────────────────────────────────

    def on_run(self):
        if self._running:
            return
        cfg = self._collect_config()
        if not cfg.use_H and not cfg.use_F:
            self._set_status('Activez au moins une méthode (H ou F).', RED)
            return

        self._running = True
        self._run_btn.configure(state='disabled', text='  Calcul…')
        self._prog.set(0)
        self._prog.pack(side='left', padx=14, pady=7)
        self._set_status('Analyse en cours…', PEACH)

        def thread():
            try:
                if cfg.mode == 'single':
                    scene = make_scene(cfg)
                    res   = run_pipeline(scene, cfg)
                    self.after(0, lambda: self._show_single(scene, res, cfg))

                elif cfg.mode == 'noise_sweep':
                    data = _noise_sweep(cfg)
                    self.after(0, lambda: self._show_noise(data, cfg))

                elif cfg.mode == 'montecarlo':
                    def _prog_cb(done, total):
                        p = done / total
                        self.after(0, lambda: (
                            self._prog.set(p),
                            self._set_status(
                                f'Monte-Carlo : {done}/{total}…', PEACH)
                        ))
                    data = _montecarlo(cfg, _prog_cb)
                    self.after(0, lambda: self._show_mc(data, cfg))

            except Exception as exc:
                self.after(0, lambda e=exc:
                           self._set_status(f'Erreur : {e}', RED))
            finally:
                self.after(0, self._done)

        threading.Thread(target=thread, daemon=True).start()

    def _done(self):
        self._running = False
        self._run_btn.configure(state='normal', text='  Analyser')
        self._prog.pack_forget()

    def _set_status(self, msg, color=SUBTEXT):
        self._status.configure(text=msg, text_color=color)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _style_ax2d(self, ax):
        ax.set_facecolor('#181825')

    def _style_ax3d(self, ax):
        ax.set_facecolor('#181825')
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#45475a')
        ax.grid(True, color='#313244', alpha=0.5)

    def _show_single(self, scene, res, cfg):
        self._fig.clear()
        gs = self._fig.add_gridspec(
            4, 2, width_ratios=[2.2, 1.0],
            height_ratios=[1, 1, 1, 1],
            hspace=0.60, wspace=0.28,
        )
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

        scene_lbl = 'Planaire' if cfg.scene_type == 'planar' else 'Non-planaire'
        self._fig.suptitle(
            f'Scène {scene_lbl}  ·  {res["n_visible"]} pts visibles  ·  '
            f'Vainqueur [{win}]',
            fontsize=12, fontweight='bold', color=TEXT, y=0.995)

        self._mpl_canvas.draw_idle()
        self._export_if_needed(cfg, 'single')

        # Results card
        lines = [f'Points visibles : {res["n_visible"]}']
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
            f'Analyse terminée  ·  {res["n_visible"]} pts  ·  [{win}]', GREEN)

    def _show_noise(self, data, cfg):
        self._fig.clear()
        fig = self._fig

        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self._style_ax2d(ax1)
        self._style_ax2d(ax2)

        levels = [d['sigma'] for d in data]
        H_C, F_C = '#fab387', '#f38ba8'

        for ax, key_H, key_F, ylabel, title in [
            (ax1, 'err_R_H', 'err_R_F',
             'Erreur rotation (°)', 'Rotation'),
            (ax2, 'err_t_H', 'err_t_F',
             'Erreur direction t (°)', 'Translation'),
        ]:
            if cfg.use_H:
                ax.plot(levels, [d[key_H] for d in data],
                        'o-', color=H_C, lw=2, ms=6, label='H')
            if cfg.use_F:
                ax.plot(levels, [d[key_F] for d in data],
                        's-', color=F_C, lw=2, ms=6, label='F')
            ax.set_xlabel('σ  (px)')
            ax.set_ylabel(ylabel)
            ax.set_title(title, color=TEXT)
            ax.legend()
            ax.grid(True, alpha=0.25)

        fig.suptitle('Robustesse au bruit', fontsize=13,
                     fontweight='bold', color=TEXT)
        self._mpl_canvas.draw_idle()
        self._export_if_needed(cfg, 'noise_sweep')
        self._update_res_box(
            f'Balayage : {len(levels)} niveaux\n'
            f'σ de {min(levels):.1f} à {max(levels):.1f} px')
        self._set_status('Balayage bruit terminé.', GREEN)

    def _show_mc(self, data, cfg):
        self._fig.clear()
        fig = self._fig

        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self._style_ax2d(ax1)
        self._style_ax2d(ax2)

        H_C, F_C = '#fab387', '#f38ba8'

        def _cl(arr):
            a = np.array(arr, dtype=float)
            return a[~np.isnan(a)]

        lines = [f'Monte-Carlo : {len(data)} scènes']

        for ax, kH, kF, ylabel, title in [
            (ax1, 'err_R_H', 'err_R_F',
             'Erreur rotation (°)', 'Rotation'),
            (ax2, 'err_t_H', 'err_t_F',
             'Erreur direction t (°)', 'Translation'),
        ]:
            boxes, labels, colors = [], [], []
            if cfg.use_H:
                d = _cl([r[kH] for r in data])
                if len(d):
                    boxes.append(d); labels.append('H'); colors.append(H_C)
                    lines.append(
                        f'{kH}: μ={d.mean():.3f}°  σ={d.std():.3f}°')
            if cfg.use_F:
                d = _cl([r[kF] for r in data])
                if len(d):
                    boxes.append(d); labels.append('F'); colors.append(F_C)
                    lines.append(
                        f'{kF}: μ={d.mean():.3f}°  σ={d.std():.3f}°')
            if boxes:
                bp = ax.boxplot(boxes, patch_artist=True, widths=0.45,
                                medianprops=dict(color='white', lw=2))
                for patch, col in zip(bp['boxes'], colors):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.75)
                for el in ['whiskers', 'caps', 'fliers']:
                    for item in bp[el]:
                        item.set_color('#585b70')
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title, color=TEXT)
            ax.grid(True, axis='y', alpha=0.25)

        fig.suptitle(f'Monte-Carlo  ({len(data)} scènes)',
                     fontsize=13, fontweight='bold', color=TEXT)
        self._mpl_canvas.draw_idle()
        self._export_if_needed(cfg, 'montecarlo')
        self._update_res_box('\n'.join(lines))
        self._set_status(f'Monte-Carlo terminé  ·  {len(data)} scènes.', GREEN)

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
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = os.path.join(TOOL_DIR, f'result_{suffix}_{ts}.png')
        self._fig.savefig(out, dpi=150, bbox_inches='tight',
                          facecolor=self._fig.get_facecolor())
        self._set_status(f'Figure sauvegardée : {os.path.basename(out)}', TEAL)

    def _reset(self):
        defaults = Config()
        self._scene_type_var.set('Planaire')
        self._z_min.set(defaults.z_min)
        self._z_max.set(defaults.z_max)
        self._x_range.set(defaults.x_range)
        self._y_range.set(defaults.y_range)
        self._n_pts.set(defaults.n_points)
        self._noise.set(defaults.noise_sigma)
        self._outl.set(defaults.outlier_ratio * 100)
        self._seed.set(defaults.seed)
        self._c1_rot.set(defaults.cam1_rx, defaults.cam1_ry, defaults.cam1_rz)
        self._c1_trans.set(defaults.cam1_tx, defaults.cam1_ty, defaults.cam1_tz)
        self._c2_rot.set(defaults.cam2_rx, defaults.cam2_ry, defaults.cam2_rz)
        self._c2_trans.set(defaults.cam2_tx, defaults.cam2_ty, defaults.cam2_tz)
        self._fx.set(defaults.fx); self._fy.set(defaults.fy)
        self._cx.set(defaults.cx); self._cy.set(defaults.cy)
        self._img_w.set(defaults.img_w); self._img_h.set(defaults.img_h)
        self._use_h.select(); self._use_f.select()
        self._mode_var.set('Scene unique')
        self._on_mode('Scene unique')
        self._update_fov()
        self._show_placeholder()
        self._set_status('Paramètres réinitialisés.', SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
