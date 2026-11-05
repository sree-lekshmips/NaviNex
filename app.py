"""
VisionAssist 2.0 - Desktop Application Shell

Modern, colourful and animated accessibility dashboard, built in pure Tkinter.
Three switchable colour themes (click the colour dots in the header):

    * Aurora   - deep violet background with purple -> magenta -> pink accents
    * Blossom  - soft blush light mode with pink -> purple -> indigo accents
    * Emerald  - deep forest background with green -> teal -> lime accents

Animated elements: flowing gradient title and header bar, rotating gradient
logo ring with pulsing sound waves, smooth card hover transitions, floating
glow orbs and pulse rings in the preview placeholder, and a pulsing status
dot. Animations pause automatically while the camera/video is running so the
AI pipeline gets all of the CPU.

Contains real-time AI computer vision for blind navigation assistance:
SegFormer walkable path segmentation, MiDaS depth estimation, YOLO obstacle
detection, and natural voice guidance. (No external image assets.)
"""
from typing import Optional
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from datetime import datetime
import cv2
import time
from PIL import Image, ImageTk
from config import SEGFORMER_MODEL_DIR
from src.detection import CustomYOLODetector, CocoYOLODetector
from src.navigation import NavigationEngine
from src.depth import MiDaSDepthEstimator
from src.segmentation import SegFormerSegmenter
from src.guidance import GuidanceEngine, ObstacleInfo
from src.voice import VoiceGuide
import threading
import os
from src.feature_logger import FeatureLogger

# ==== Colour themes ====
# stops:    3-colour gradient used for the title, header bar, logo ring and swatches
# accent_a: accent of the "Live Webcam" card      accent_b: accent of the "Upload Video" card

THEMES = {
    
    "Emerald": {
        "dark": True,
        "bg": "#04130F", "surface": "#0A2019", "surface_hi": "#10332A", "border": "#1B5546",
        "text": "#ECFDF5", "muted": "#8FBFAE",
        "stops": ["#10B981", "#2DD4BF", "#84C781"],
        "accent_a": "#34D399", "accent_b": "#2DD4BF",
        "ok": "#4ADE80", "warn": "#FACC15", "bad": "#F87171",
    },
}
DEFAULT_THEME: str = "Emerald"


def _clamp(value: int) -> int:
    """Clamp an int color channel into the valid 0-255 range."""
    return max(0, min(255, value))


def _blend(color_a: str, color_b: str, t: float) -> str:
    """Linearly blend two '#rrggbb' colors.

    Args:
        color_a (str): Start color (t=0).
        color_b (str): End color (t=1).
        t (float): Blend factor between 0 and 1.

    Returns:
        str: The blended color as a '#rrggbb' hex string.
    """
    a = tuple(int(color_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(_clamp(int(a[i] + (b[i] - a[i]) * t)) for i in range(3))
    return "#%02x%02x%02x" % mixed


class VisionAssistApp:
    """Main application window for NaviNex.

    Builds and manages the primary GUI: an animated gradient logo, a flowing
    gradient title, a theme switcher and safety badge, two accent-coloured
    mode-selection cards, an animated camera/video preview panel and a status
    card with a pulsing indicator.
    """

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application window and build all UI elements.

        Args:
            root (tk.Tk): The root tkinter window to build the UI into.
        """
        self.root: tk.Tk = root
        self.cap = None
        self.camera_running = False

        # UI-only state (theme, animation clock, widget handles)
        self.theme_name: str = DEFAULT_THEME
        self.theme: dict = THEMES[self.theme_name]
        self._tick: int = 0
        self._preview_active = None
        self._last_status_key = None
        self._status_color: str = self.theme["ok"]
        self._status_dot = None
        self._status_border_id = None
        self._title_letters = []
        self._logo_items = None
        self._ring_items = None
        self._bar_canvas = None
        self._bar_items = []
        self._orbs = []
        self._live_items = []
        self.status_label = None
        self.preview_label = None

        # Load VisionAssist AI core modules
        self.custom_detector = CustomYOLODetector()
        self.coco_detector = CocoYOLODetector()
        self.depth_estimator = MiDaSDepthEstimator()
        self.navigator = NavigationEngine()
        self.feature_logger = FeatureLogger("outputs/features.csv")
        self.video_path = "webcam"
        self.guidance_engine = GuidanceEngine(frame_width=1280)
        self.voice = VoiceGuide(cooldown_seconds=2.8)

        # Load trained SegFormer segmenter
        self.segmenter = None
        try:
            if SEGFORMER_MODEL_DIR.exists():
                self.segmenter = SegFormerSegmenter()
                print(f"[SegFormer] Loaded fine-tuned model from '{SEGFORMER_MODEL_DIR}'.")
        except Exception as e:
            print(f"[SegFormer Warning] Failed to initialize SegFormer: {e}")
            self.segmenter = None

        self.last_spoken_command = ""
        self.current_instruction = "System ready."
        self._configure_window()
        self._build_fonts()

        # Build UI sections top to bottom
        self._build_ui()

        # Performance caching variables
        self.frame_count = 0
        self.cached_custom_results = None
        self.cached_coco_results = None
        self.cached_depth = None
        self.cached_seg_map = None
        self.cached_walkable_mask = None

        # Start the (lightweight) UI animation loop
        self.root.after(60, self._animate)

    # --- Setup ---------------------------------------------------------------

    def _configure_window(self) -> None:
        """Set the window title, size, and base theme background."""
        self.root.title("NaviNex")
        self.root.geometry("1050x700")
        self.root.configure(bg=self.theme["bg"])
        self.root.minsize(950, 650)

    def _build_fonts(self) -> None:
        """Create the named fonts used throughout the UI."""
        self._title_font = tkfont.Font(family="Segoe UI", size=32, weight="bold")
        self._subtitle_font = tkfont.Font(family="Segoe UI", size=12)
        self._badge_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self._card_title_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self._card_subtitle_font = tkfont.Font(family="Segoe UI", size=10)
        self._preview_title_font = tkfont.Font(family="Segoe UI", size=17, weight="bold")
        self._preview_subtitle_font = tkfont.Font(family="Segoe UI", size=10)
        self._status_title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self._status_subtitle_font = tkfont.Font(family="Segoe UI", size=9)
        self._live_font = tkfont.Font(family="Segoe UI", size=8, weight="bold")

    def _build_ui(self) -> None:
        """Build (or rebuild) every UI section for the current theme."""
        self._build_header()
        self._build_gradient_bar()
        self._build_action_cards()
        self._build_preview_panel()
        self._build_status_bar()

    def set_theme(self, name: str) -> None:
        """Switch colour theme at runtime (UI only - the AI pipeline is untouched)."""
        if name not in THEMES or name == self.theme_name:
            return
        status_text = self.status_label.cget("text") if self.status_label is not None else None
        self.theme_name = name
        self.theme = THEMES[name]
        self.root.configure(bg=self.theme["bg"])
        for child in self.root.winfo_children():
            child.destroy()
        self._title_letters, self._logo_items, self._ring_items = [], None, None
        self._bar_items, self._orbs, self._live_items = [], [], []
        self._status_dot, self._status_border_id = None, None
        self._preview_active, self._last_status_key = None, None
        self._build_ui()
        if status_text is not None:
            self.status_label.config(text=status_text)

    # --- Colour / drawing helpers ---------------------------------------------

    def _grad(self, u: float, cyclic: bool = False) -> str:
        """Sample the theme's 3-stop gradient at position u (0..1, wraps if cyclic)."""
        stops = list(self.theme["stops"])
        if cyclic:
            stops.append(stops[0])
            u = u % 1.0
        else:
            u = max(0.0, min(1.0, u))
        pos = u * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        return _blend(stops[i], stops[i + 1], pos - i)

    @staticmethod
    def _rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius=20, **kwargs) -> int:
        """Draw a rounded rectangle (as a smoothed polygon) and return its id."""
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1,
            x2, y1 + radius, x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2, x1, y2,
            x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _card_backdrop(self, canvas, x1, y1, x2, y2, radius, accent, tags=None) -> None:
        """Soft glow (dark themes) or soft shadow (light theme) behind a card."""
        th = self.theme
        tag_kwargs = {"tags": tags} if tags else {}
        if th["dark"]:
            for i, strength in ((6, 0.07), (3, 0.14)):
                self._rounded_rect(
                    canvas, x1 - i, y1 - i, x2 + i, y2 + i, radius=radius + i,
                    fill=_blend(th["bg"], accent, strength), outline="", **tag_kwargs,
                )
        else:
            for off, strength in ((4, 0.05), (2, 0.10)):
                self._rounded_rect(
                    canvas, x1, y1 + off, x2, y2 + off, radius=radius,
                    fill=_blend(th["bg"], th["border"], strength * 6), outline="", **tag_kwargs,
                )

    # ==== Header & Logo ====

    def _build_logo(self, parent: tk.Widget) -> tk.Canvas:
        """Draw the accessibility logo: walking figure inside a rotating gradient ring."""
        th = self.theme
        size = 90
        cx, cy, r = size / 2, size / 2, 32
        canvas = tk.Canvas(parent, width=size + 30, height=size, bg=th["bg"], highlightthickness=0)

        # Soft tinted disc behind the figure
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=_blend(th["bg"], th["stops"][0], 0.14), outline="")

        # Gradient ring made of short arcs (their colours rotate in _animate)
        arcs = []
        segments = 12
        for i in range(segments):
            arcs.append(canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r, start=i * (360 / segments), extent=360 / segments + 1,
                style=tk.ARC, outline=self._grad(i / segments, cyclic=True), width=3,
            ))

        # Walking person with a cane, centered in the ring
        fig = th["text"]
        canvas.create_oval(cx - 4, cy - 18, cx + 4, cy - 10, fill=fig, outline="")  # head
        canvas.create_line(cx, cy - 10, cx, cy + 6, fill=fig, width=3)              # torso
        canvas.create_line(cx, cy - 4, cx - 8, cy - 8, fill=fig, width=3)           # back arm
        canvas.create_line(cx, cy - 4, cx + 6, cy + 2, fill=fig, width=3)           # cane-holding arm
        canvas.create_line(cx, cy + 6, cx - 7, cy + 18, fill=fig, width=3)          # back leg
        canvas.create_line(cx, cy + 6, cx + 5, cy + 18, fill=fig, width=3)          # front leg
        canvas.create_line(cx + 6, cy + 2, cx + 12, cy + 20, fill=th["accent_b"], width=2)  # cane

        # Sound-wave arcs to the right of the ring (pulse in _animate)
        waves = []
        wave_x = cx + r + 14
        for rad in (6, 11, 16):
            waves.append(canvas.create_arc(
                wave_x - rad, cy - rad, wave_x + rad, cy + rad,
                start=300, extent=120, style=tk.ARC, outline=th["accent_b"], width=2,
            ))

        self._logo_items = (canvas, arcs, waves)
        return canvas

    def _build_theme_swatches(self, parent: tk.Widget) -> tk.Canvas:
        """Three clickable gradient dots used to switch theme."""
        th = self.theme
        names = list(THEMES.keys())
        canvas = tk.Canvas(parent, width=len(names) * 30 + 6, height=34, bg=th["bg"], highlightthickness=0)
        for idx, name in enumerate(names):
            tag = f"swatch_{name}"
            cx, cy, r = 19 + idx * 30, 17, 9
            stops = THEMES[name]["stops"]
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, extent=180, style=tk.PIESLICE,
                              fill=stops[0], outline="", tags=tag)
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=270, extent=180, style=tk.PIESLICE,
                              fill=stops[2], outline="", tags=tag)
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=stops[1], outline="", tags=tag)
            if name == self.theme_name:
                canvas.create_oval(cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, outline=th["text"], width=2)
            canvas.tag_bind(tag, "<Button-1>", lambda _e, n=name: self.root.after(1, lambda: self.set_theme(n)))
            canvas.tag_bind(tag, "<Enter>", lambda _e: canvas.configure(cursor="hand2"))
            canvas.tag_bind(tag, "<Leave>", lambda _e: canvas.configure(cursor=""))
        return canvas

    def _build_header(self) -> None:
        """Build the logo, gradient title/subtitle, theme switcher and safety badge."""
        th = self.theme
        header_frame = tk.Frame(self.root, bg=th["bg"])
        header_frame.pack(fill=tk.X, padx=30, pady=(18, 6))

        self._build_logo(header_frame).pack(side=tk.LEFT, anchor="n", pady=4)

        # Safety badge, top-right: soft tinted pill with a coloured outline
        badge_canvas = tk.Canvas(header_frame, width=210, height=34, bg=th["bg"], highlightthickness=0)
        badge_canvas.pack(side=tk.RIGHT, anchor="n")
        self._rounded_rect(
            badge_canvas, 2, 2, 208, 32, radius=16,
            fill=_blend(th["bg"], th["accent_b"], 0.12), outline=th["accent_b"], width=1,
        )
        badge_canvas.create_text(105, 17, text="🛡  Safe. Smart. Independent.", fill=th["accent_b"], font=self._badge_font)

        # Theme switcher (UI only), just left of the badge
        self._build_theme_swatches(header_frame).pack(side=tk.RIGHT, anchor="n", padx=(0, 12))

        # Centered gradient title (one label per letter so the colours can flow) + subtitle
        title_frame = tk.Frame(header_frame, bg=th["bg"])
        title_frame.pack(expand=True)

        title_row = tk.Frame(title_frame, bg=th["bg"])
        title_row.pack()
        self._title_letters = []
        letters = ["N", "a", "v", "i", "N", "e", "X", "👁️"]
        for idx, chunk in enumerate(letters):
            lbl = tk.Label(title_row, text=chunk, font=self._title_font, fg=self._grad(idx / len(letters)), bg=th["bg"])
            lbl.pack(side=tk.LEFT, padx=(0, 8) if idx == 3 else (0, 0))
            self._title_letters.append(lbl)

        tk.Label(
            title_frame, text="Blind Navigation Assistant",
            font=self._subtitle_font, fg=th["muted"], bg=th["bg"],
        ).pack(side=tk.LEFT)

    def _build_gradient_bar(self) -> None:
        """A thin animated gradient line under the header."""
        self._bar_canvas = tk.Canvas(self.root, height=4, bg=self.theme["bg"], highlightthickness=0)
        self._bar_canvas.pack(fill=tk.X, padx=30, pady=(2, 6))
        self._bar_canvas.bind("<Configure>", self._rebuild_bar)

    def _rebuild_bar(self, event: tk.Event) -> None:
        """Recreate the gradient segments to fit the new width."""
        c = self._bar_canvas
        c.delete("all")
        n = max(20, event.width // 6)
        seg = event.width / n
        self._bar_items = [
            c.create_rectangle(i * seg, 0, (i + 1) * seg + 1, 4, fill=self._grad(i / n, cyclic=True), width=0)
            for i in range(n)
        ]

    # ==== Action cards ====

    def _build_action_cards(self) -> None:
        """Build the two large, clickable action cards (each with its own accent colour)."""
        container = tk.Frame(self.root, bg=self.theme["bg"])
        container.pack(pady=(8, 14))

        self._make_action_card(
            parent=container, icon="🎥", title="Live Webcam", subtitle="Real-time Detection",
            accent=self.theme["accent_a"], command=self.start_webcam,
        ).grid(row=0, column=0, padx=12)

        self._make_action_card(
            parent=container, icon="📁", title="Upload Video", subtitle="Analyze Recorded Video",
            accent=self.theme["accent_b"], command=self.upload_video,
        ).grid(row=0, column=1, padx=12)

    def _make_action_card(self, parent, icon, title, subtitle, accent, command, width=440, height=86) -> tk.Canvas:
        """Create a single rounded action card with an animated hover transition.

        Args:
            parent (tk.Widget): Parent widget to place the card in.
            icon (str): Emoji/icon shown inside the circular icon badge.
            title (str): Bold primary label.
            subtitle (str): Smaller secondary label.
            accent (str): Accent colour used for the border, icon and chevron.
            command: Callback invoked when the card is clicked.

        Returns:
            tk.Canvas: The canvas widget representing the card.
        """
        th = self.theme
        canvas = tk.Canvas(parent, width=width, height=height, bg=th["bg"], highlightthickness=0)

        # Hover halo (invisible until hover), backdrop, then the card body
        halo_id = self._rounded_rect(canvas, 4, 4, width - 4, height - 4, radius=20, fill="", outline=th["bg"], width=3)
        self._card_backdrop(canvas, 8, 8, width - 8, height - 8, 16, accent)
        body_id = self._rounded_rect(
            canvas, 8, 8, width - 8, height - 8, radius=16,
            fill=th["surface"], outline=th["border"], width=1,
        )

        # Circular icon badge tinted with the accent
        icon_r = 24
        icon_cx, icon_cy = 50, height // 2
        canvas.create_oval(
            icon_cx - icon_r, icon_cy - icon_r, icon_cx + icon_r, icon_cy + icon_r,
            fill=_blend(th["surface"], accent, 0.22), outline=_blend(th["surface"], accent, 0.6), width=2,
        )
        canvas.create_text(icon_cx, icon_cy, text=icon, font=("Segoe UI Emoji", 18))

        # Title + subtitle
        canvas.create_text(90, height // 2 - 12, text=title, anchor="w", fill=th["text"], font=self._card_title_font)
        canvas.create_text(90, height // 2 + 12, text=subtitle, anchor="w", fill=th["muted"], font=self._card_subtitle_font)

        # Trailing chevron (slides right on hover)
        chevron_id = canvas.create_text(width - 30, height // 2, text="›", fill=accent, font=("Segoe UI", 20, "bold"))

        # Smooth hover transition: v animates 0 -> 1 on Enter, 1 -> 0 on Leave
        anim = {"v": 0.0, "target": 0.0, "job": None}

        def apply() -> None:
            v = anim["v"]
            canvas.itemconfig(body_id, fill=_blend(th["surface"], accent, 0.14 * v),
                              outline=_blend(th["border"], accent, v))
            canvas.itemconfig(halo_id, outline=_blend(th["bg"], accent, 0.45 * v))
            canvas.coords(chevron_id, width - 30 + 5 * v, height // 2)

        def step() -> None:
            try:
                diff = anim["target"] - anim["v"]
                if abs(diff) < 0.04:
                    anim["v"], anim["job"] = anim["target"], None
                    apply()
                    return
                anim["v"] += diff * 0.35
                apply()
                anim["job"] = canvas.after(16, step)
            except tk.TclError:
                pass  # card was destroyed (e.g. theme switch)

        def go(target: float) -> None:
            anim["target"] = target
            if anim["job"] is None:
                step()

        canvas.bind("<Enter>", lambda _e: go(1.0))
        canvas.bind("<Leave>", lambda _e: go(0.0))
        canvas.configure(cursor="hand2")
        canvas.bind("<Button-1>", lambda _e: command())

        return canvas

    # ==== Preview panel ====

    def _build_preview_panel(self) -> None:
        """Build the animated camera/video preview panel."""
        th = self.theme
        outer = tk.Frame(self.root, bg=th["bg"])
        outer.pack(padx=30, pady=6, fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(outer, bg=th["bg"], highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self._content_frame = tk.Frame(self.preview_canvas, bg=th["surface"])
        content = self._content_frame
        self.preview_canvas.create_window(0, 0, window=content, anchor="center", tags="preview_content")

        # Circular camera placeholder: rotating gradient ring + pulse rings
        ring_canvas = tk.Canvas(content, width=110, height=110, bg=th["surface"], highlightthickness=0)
        ring_canvas.pack(pady=(0, 10))
        pulses = [
            ring_canvas.create_oval(11, 11, 99, 99, outline=th["surface"], width=2) for _ in range(2)
        ]
        ring_arcs = []
        for i in range(12):
            ring_arcs.append(ring_canvas.create_arc(
                15, 15, 95, 95, start=i * 30, extent=31, style=tk.ARC,
                outline=self._grad(i / 12, cyclic=True), width=4,
            ))
        ring_canvas.create_oval(24, 24, 86, 86, fill=th["surface_hi"], outline="")
        ring_canvas.create_text(55, 55, text="📷", font=("Segoe UI Emoji", 26))
        self._ring_items = (ring_canvas, ring_arcs, pulses)

        self.preview_label: tk.Label = tk.Label(
            content, text="Camera / Video Preview", font=self._preview_title_font, fg=th["text"], bg=th["surface"],
        )
        self.preview_label.pack()

        tk.Label(
            content, text="Your live feed or uploaded video will appear here",
            font=self._preview_subtitle_font, fg=th["muted"], bg=th["surface"],
        ).pack(pady=(4, 0))

        self.preview_canvas.bind("<Configure>", self._draw_preview_decorations)

    def _draw_preview_decorations(self, event: Optional[tk.Event] = None) -> None:
        """Redraw the preview card: backdrop, gradient border, corner brackets, glow orbs, ACTIVE pill."""
        th = self.theme
        c = self.preview_canvas
        w = event.width if event is not None else c.winfo_width()
        h = event.height if event is not None else c.winfo_height()
        if w <= 1 or h <= 1:
            return
        c.delete("deco")
        c.delete("orb")
        c.delete("live")
        active = bool(self._preview_active)

        # Backdrop + card body + border
        self._card_backdrop(c, 4, 4, w - 4, h - 4, 24, th["accent_b"], tags="deco")
        self._rounded_rect(
            c, 4, 4, w - 4, h - 4, radius=24, fill=th["surface"],
            outline=th["accent_a"] if active else _blend(th["border"], th["accent_b"], 0.35),
            width=2, tags="deco",
        )

        # Gradient corner brackets (viewfinder feel)
        bracket = 28
        for idx, (bx, by, dx, dy) in enumerate((
            (20, 20, 1, 1), (w - 20, 20, -1, 1), (20, h - 20, 1, -1), (w - 20, h - 20, -1, -1),
        )):
            c.create_line(bx, by + bracket * dy, bx, by, bx + bracket * dx, by,
                          fill=self._grad(idx / 3), width=3, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="deco")

        # Floating glow orbs (left and right of the placeholder content)
        self._orbs = []
        content_w = self._content_frame.winfo_reqwidth() + 24
        zones = ((26, w / 2 - content_w / 2 - 10), (w / 2 + content_w / 2 + 10, w - 26))
        specs = (
            (0, 1.0, th["accent_a"], 0.0, 0.7), (0, 0.5, th["accent_b"], 2.1, 1.0),
            (1, 1.0, th["accent_b"], 1.3, 0.8), (1, 0.5, th["accent_a"], 3.4, 1.1),
        )
        for zone_idx, rfac, color, phase, speed in specs:
            x0, x1 = zones[zone_idx]
            r = min(64.0, (x1 - x0) / 2 - 4) * rfac
            if r < 14:
                continue
            ids = []
            for k in range(6):
                ids.append(c.create_oval(0, 0, 1, 1, fill=_blend(th["surface"], color, 0.05 + 0.05 * k),
                                         outline="", tags="orb"))
            self._orbs.append({"ids": ids, "r": r, "x0": x0, "x1": x1, "phase": phase, "speed": speed})
        self._place_orbs(self._tick * 0.05)

        # "ACTIVE" pill while a camera/video is running
        if active:
            pill = self._rounded_rect(c, 40, 34, 112, 58, radius=12, fill=_blend(th["surface"], th["ok"], 0.18),
                                      outline=th["ok"], width=1, tags="live")
            dot = c.create_oval(50, 42, 58, 50, fill=th["ok"], outline="", tags="live")
            txt = c.create_text(66, 46, text="ACTIVE", anchor="w", fill=th["ok"], font=self._live_font, tags="live")

        # Keep the placeholder content above the decorations
        c.coords("preview_content", w // 2, h // 2)
        c.tag_raise("orb")
        c.tag_raise("live")
        c.tag_raise("preview_content")
        c.itemconfig("orb", state="hidden" if self.camera_running else "normal")

    def _place_orbs(self, t: float) -> None:
        """Move every orb along a smooth looping path."""
        c = self.preview_canvas
        h = c.winfo_height()
        for orb in self._orbs:
            r, x0, x1 = orb["r"], orb["x0"], orb["x1"]
            cx = (x0 + x1) / 2 + ((x1 - x0) / 2 - r) * 0.85 * math.sin(t * orb["speed"] + orb["phase"])
            cy = h / 2 + (h / 2 - r - 24) * 0.85 * math.sin(t * orb["speed"] * 0.8 + orb["phase"] * 1.7)
            for k, oid in enumerate(orb["ids"]):
                rad = r * (1 - k * 0.13)
                c.coords(oid, cx - rad, cy - rad, cx + rad, cy + rad)

    # ==== Status bar ====

    def _build_status_bar(self) -> None:
        """Build a single centered 'Status: Ready' card."""
        bar = tk.Frame(self.root, bg=self.theme["bg"])
        bar.pack(fill=tk.X, padx=30, pady=(6, 20))

        status_canvas, self.status_label = self._make_status_card(
            bar, icon="🟢", icon_color=self.theme["ok"], glow_color=self.theme["ok"],
            title="Status: Ready", subtitle="System is ready to assist you",
            width=340,
        )
        status_canvas.pack(pady=0)

    def _make_status_card(self, parent, icon, icon_color, glow_color, title, subtitle, width=180):
        """Create a rounded status card with a pulsing coloured status dot.

        'icon' and 'glow_color' are kept for API compatibility; the card uses a
        pulsing dot (icon_color) in place of an emoji.
        """
        th = self.theme
        height = 64
        canvas = tk.Canvas(parent, width=width, height=height, bg=th["bg"], highlightthickness=0)

        def draw(event: Optional[tk.Event] = None) -> None:
            canvas.delete("card_bg")
            w = canvas.winfo_width() or 180
            self._card_backdrop(canvas, 4, 4, w - 4, height - 6, 16, self._status_color, tags="card_bg")
            self._status_border_id = self._rounded_rect(
                canvas, 4, 4, w - 4, height - 6, radius=16, fill=th["surface"],
                outline=_blend(th["border"], self._status_color, 0.7), width=2, tags="card_bg",
            )
            canvas.tag_raise("icon_chip")
            canvas.tag_raise("labels")

        canvas.bind("<Configure>", draw)

        # Pulsing status dot
        holder_size = 40
        icon_holder = tk.Canvas(canvas, width=holder_size, height=holder_size, bg=th["surface"], highlightthickness=0)
        pulse_id = icon_holder.create_oval(10, 10, 30, 30, outline=icon_color, width=2)
        halo_id = icon_holder.create_oval(8, 8, 32, 32, fill=_blend(th["surface"], icon_color, 0.2), outline="")
        dot_id = icon_holder.create_oval(13, 13, 27, 27, fill=icon_color, outline="")
        canvas.create_window(34, (height - 3) // 2, window=icon_holder, tags="icon_chip")
        self._status_dot = (icon_holder, halo_id, dot_id, pulse_id)
        self._status_color = icon_color

        title_label = tk.Label(canvas, text=title, font=self._status_title_font, fg=th["text"], bg=th["surface"], anchor="w")
        canvas.create_window(62, (height - 3) // 2 - 10, window=title_label, anchor="w", tags="labels")

        subtitle_label = tk.Label(canvas, text=subtitle, font=self._status_subtitle_font, fg=th["muted"], bg=th["surface"], anchor="w")
        canvas.create_window(62, (height - 3) // 2 + 10, window=subtitle_label, anchor="w", tags="labels")

        self._status_canvas = canvas
        return canvas, title_label

    # --- UI-only animation & refresh (colours / positions only; no pipeline logic) ---

    def _animate(self) -> None:
        """Drive all decorative animations. Heavy ones pause while video is running."""
        try:
            self._tick += 1
            t = self._tick * 0.06
            running = bool(self.camera_running)

            if self._tick % 4 == 0:
                self._sync_ui_state()

            self._anim_status(t)
            if not running:
                self._anim_title(t)
                self._anim_logo(t)
                self._anim_ring(t)
                if self._tick % 2 == 0:
                    self._anim_bar(t)
                self._place_orbs(t)
        except tk.TclError:
            pass  # widgets were rebuilt / destroyed mid-tick
        try:
            self.root.after(60, self._animate)
        except tk.TclError:
            return  # window closed

    def _anim_title(self, t: float) -> None:
        """Flow the gradient across the title letters."""
        n = len(self._title_letters)
        for i, lbl in enumerate(self._title_letters):
            lbl.config(fg=self._grad(i / max(1, n) * 0.9 - t * 0.08, cyclic=True))

    def _anim_logo(self, t: float) -> None:
        """Rotate the logo ring colours and pulse the sound waves."""
        if not self._logo_items:
            return
        canvas, arcs, waves = self._logo_items
        n = len(arcs)
        for i, aid in enumerate(arcs):
            canvas.itemconfig(aid, outline=self._grad(i / n - t * 0.12, cyclic=True))
        for i, wid in enumerate(waves):
            k = (math.sin(t * 2.2 - i * 0.9) + 1) / 2
            canvas.itemconfig(wid, outline=_blend(self.theme["accent_b"], self.theme["bg"], 0.15 + 0.7 * (1 - k)))

    def _anim_ring(self, t: float) -> None:
        """Rotate the camera ring colours and expand soft pulse rings."""
        if not self._ring_items:
            return
        canvas, arcs, pulses = self._ring_items
        n = len(arcs)
        for i, aid in enumerate(arcs):
            canvas.itemconfig(aid, outline=self._grad(i / n - t * 0.1, cyclic=True))
        for j, pid in enumerate(pulses):
            p = (t * 0.45 + j * 0.5) % 1.0
            rad = 44 + 10 * p
            canvas.coords(pid, 55 - rad, 55 - rad, 55 + rad, 55 + rad)
            canvas.itemconfig(pid, outline=_blend(self.theme["accent_b"], self.theme["surface"], 0.25 + 0.75 * p))

    def _anim_bar(self, t: float) -> None:
        """Flow the gradient along the header bar."""
        n = len(self._bar_items)
        if not n:
            return
        for i, rid in enumerate(self._bar_items):
            self._bar_canvas.itemconfig(rid, fill=self._grad(i / n - t * 0.06, cyclic=True))

    def _anim_status(self, t: float) -> None:
        """Pulse ring around the status dot."""
        if self._status_dot is None:
            return
        holder, _halo, _dot, pulse = self._status_dot
        p = (t * 0.55) % 1.0
        rad = 7 + 11 * p
        holder.coords(pulse, 20 - rad, 20 - rad, 20 + rad, 20 + rad)
        holder.itemconfig(pulse, outline=_blend(self._status_color, self.theme["surface"], 0.2 + 0.8 * p))

    def _sync_ui_state(self) -> None:
        """Refresh visual state derived from the app: preview accent + status colour."""
        th = self.theme
        active = bool(self.camera_running)
        if active != self._preview_active:
            self._preview_active = active
            self._draw_preview_decorations()

        if self.status_label is None or self._status_dot is None:
            return
        text = self.status_label.cget("text")
        if text != self._last_status_key:
            self._last_status_key = text
            lowered = text.lower()
            if "not found" in lowered or "unable" in lowered:
                color = th["bad"]
            elif "caution" in lowered:
                color = th["warn"]
            elif "stopped" in lowered or "finished" in lowered:
                color = th["muted"]
            else:
                color = th["ok"]
            self._status_color = color
            holder, halo_id, dot_id, _pulse = self._status_dot
            holder.itemconfig(halo_id, fill=_blend(th["surface"], color, 0.2))
            holder.itemconfig(dot_id, fill=color)
            if self._status_border_id is not None:
                self._status_canvas.itemconfig(self._status_border_id, outline=_blend(th["border"], color, 0.7))

    # --- Video & Camera Callbacks ------------------------------------------------

    def start_webcam(self) -> None:
        """Toggle webcam ON/OFF."""
        if self.camera_running:
            self.stop_webcam()
            return

        self.cap = cv2.VideoCapture(0)


        if not self.cap.isOpened():
            self.status_label.config(text="Status: Camera not found")
            return

        self.camera_running = True
        self.status_label.config(text="Status: Live Webcam Running")
        self.update_frame()

    def update_frame(self) -> None:
        """Read webcam/video frame, run multimodal AI pipeline, and display results."""
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.stop_webcam()
            self.status_label.config(text="Status: Video finished")
            return

        # -----------------------------
        # Real-time Multi-Modal AI Pipeline
        # -----------------------------
        self.frame_count += 1

        # 1. SegFormer Walkable Floor Segmentation (runs every 3 frames)
        if self.segmenter is not None:
            if self.cached_seg_map is None or (self.frame_count % 3 == 0):
                self.cached_seg_map = self.segmenter.predict(frame)
                self.cached_walkable_mask = self.segmenter.get_walkable_mask(
                    self.cached_seg_map, depth_map=self.cached_depth
                )

            seg_map = self.cached_seg_map
            walkable_mask = self.cached_walkable_mask.copy() if self.cached_walkable_mask is not None else None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, walkable_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            seg_map = None

        # 2. MiDaS Depth Estimation (runs on frame 2, 6...)
        if self.cached_depth is None or (self.frame_count % 4 == 2):
            self.cached_depth = self.depth_estimator.predict(frame)

        depth_map = self.cached_depth

        # 3. YOLO Object Detection (runs on frame 1, 3, 5...)
        if self.cached_custom_results is None or (self.frame_count % 2 == 1):
            self.cached_custom_results = self.custom_detector.predict(frame)
            self.cached_coco_results = self.coco_detector.predict(frame)

        custom_results = self.cached_custom_results
        coco_results = self.cached_coco_results

        # 4. Obstacle Distance & Danger Extraction (Filtered relevant objects)
        detected_obstacles = []
        close_found = False
        boxes_to_draw = []

        # Suppress indoor false positives on plain walls (e.g. false sofa/bed)
        SUPPRESSED_CLASSES = {"sofa", "bed", "couch", "dining table", "tv", "refrigerator", "potted plant"}

        def process_box(box, names_dict, default_color=(0, 255, 255)):
            nonlocal close_found
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names_dict.get(cls, "Object")
            label_lower = label.lower()
            cx = (x1 + x2) // 2
            fh, fw = frame.shape[:2]

            # 1. Reject tiny noise or high ceiling detections (not in walking path)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 1500 or y2 < int(fh * 0.35):
                return

            # 2. Filter false "stairs" on bookshelves / wall racks / furniture:
            if "stair" in label_lower:
                if conf < 0.68 or y2 < int(fh * 0.52):
                    return

            # 3. Suppress low-confidence hallucinations on flat walls
            if label_lower in SUPPRESSED_CLASSES and conf < 0.68:
                return

            # Sample inner median depth
            med_depth = self.depth_estimator.get_median_depth_in_box(depth_map, x1, y1, x2, y2)
            dist_m = self.depth_estimator.estimate_distance_meters(med_depth, label)
            danger = self.depth_estimator.get_danger_level(dist_m)

            detected_obstacles.append(
                ObstacleInfo(label=label, distance=dist_m, danger=danger, center_x=cx)
            )

            # Clear obstacle footprint from the walkable floor mask
            if walkable_mask is not None and walkable_mask.shape[:2] == (fh, fw):
                foot_y1 = max(0, min(fh, y1 + int((y2 - y1) * 0.55)))
                walkable_mask[foot_y1:min(fh, y2), max(0, x1):min(fw, x2)] = 0

            if danger == "DANGER":
                close_found = True
                box_color = (0, 0, 255)  # Bright Red
                badge_text = f"{label} : {dist_m:.1f}m [DANGER]"
            elif danger == "CAUTION":
                box_color = (0, 165, 255)  # Orange
                badge_text = f"{label} : ~{dist_m:.1f}m"
            else:
                box_color = default_color
                badge_text = f"{label} : ~{dist_m:.1f}m"

            boxes_to_draw.append((x1, y1, x2, y2, box_color, badge_text, danger))

        # Process Custom YOLO boxes (e.g. stairs, specific obstacles)
        for box in custom_results.boxes:
            process_box(box, custom_results.names, default_color=(0, 255, 255))

        # Process COCO YOLO boxes (e.g. person, chair, bicycle, car, door)
        for box in coco_results.boxes:
            process_box(box, coco_results.names, default_color=(255, 140, 0))

        # Process SegFormer Wall Obstacles (forward-blocking walls only)
        if self.segmenter is not None and seg_map is not None:
            wall_obstacles = self.segmenter.extract_wall_obstacles(seg_map, depth_map=depth_map)
            for wall in wall_obstacles:
                w_dist = wall["distance"]
                w_danger = wall["danger"]
                w_cx = wall["center_x"]
                wx1, wy1, wx2, wy2 = wall["box"]

                detected_obstacles.append(
                    ObstacleInfo(label="wall", distance=w_dist, danger=w_danger, center_x=w_cx)
                )

                if w_danger == "DANGER":
                    close_found = True
                    w_color = (0, 0, 255)       # Red
                    w_badge = f"Wall : {w_dist:.1f}m [DANGER]"
                elif w_danger == "CAUTION":
                    w_color = (255, 120, 0)     # Orange
                    w_badge = f"Wall : ~{w_dist:.1f}m"
                else:
                    w_color = (200, 180, 50)    # Cyan / Blue
                    w_badge = f"Wall : ~{w_dist:.1f}m"

                boxes_to_draw.append((wx1, wy1, wx2, wy2, w_color, w_badge, w_danger))

        # Overlay green walkable floor onto frame (cleanly excluding obstacle footprints)
        if self.segmenter is not None and walkable_mask is not None:
            frame = self.segmenter.overlay(frame, seg_map, walkable_mask=walkable_mask, alpha=0.35)

        # Draw all bounding boxes and labels
        for (bx1, by1, bx2, by2, b_color, b_text, b_danger) in boxes_to_draw:
            thickness = 3 if b_danger == "DANGER" else 2
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), b_color, thickness)
            (tw, th), _ = cv2.getTextSize(b_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bg_y1 = max(0, by1 - th - 10)
            cv2.rectangle(frame, (bx1, bg_y1), (bx1 + tw + 8, by1), b_color, -1)
            t_color = (255, 255, 255) if b_danger == "DANGER" else (0, 0, 0)
            cv2.putText(frame, b_text, (bx1 + 4, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 2)

        # 5. SegFormer Walkable Corridor Navigation
        nav_dict_list = [
            {"label": o.label, "distance": o.distance, "danger": o.danger, "center_x": o.center_x}
            for o in detected_obstacles
        ]
        raw_direction = self.navigator.decide_instantaneous(walkable_mask=walkable_mask, depth_map=depth_map, detected_objects=nav_dict_list)
        direction = self.navigator.decide(walkable_mask=walkable_mask, depth_map=depth_map, detected_objects=nav_dict_list)

        # Measure dynamic forward walking corridor clearance
        corridor_clearance = self.depth_estimator.get_corridor_clearance_meters(depth_map)
        if self.video_path != "webcam":
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.feature_logger.log(self.video_path, frame_idx, walkable_mask, nav_dict_list,
                                    frame.shape[1], corridor_clearance, raw_direction, direction)
        # 6. Natural Language Guidance Instruction & State-aware Voice Cues
        display_instruction, spoken_instruction = self.guidance_engine.generate_guidance(
            objects=detected_obstacles,
            direction=direction,
            corridor_clearance=corridor_clearance
        )
        self.current_instruction = display_instruction

        # Speak instruction only when state transition or obstacle triggers speech
        if spoken_instruction:
            self.voice.speak(spoken_instruction)

        # 7. Top HUD Banner (Direction & Voice Guidance)
        hud_h, hud_w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (hud_w, 48), (10, 15, 30), -1)
        cv2.line(frame, (0, 48), (hud_w, 48), (0, 200, 255), 2)

        dir_color = (0, 255, 80) if direction == "FORWARD" else (0, 220, 255) if direction in ("LEFT", "RIGHT") else (0, 0, 255)
        dir_text = f"GO {direction}" if direction != "STOP" else "STOP - BLOCKED"
        cv2.putText(frame, dir_text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.80, dir_color, 2)

        hud_speech = display_instruction if len(display_instruction) < 55 else display_instruction[:52] + "..."
        cv2.putText(frame, f'"{hud_speech}"', (270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 240), 2)

        # 8. Update status card
        if direction == "STOP" or close_found:
            self.status_label.config(text=f"Status: ⚠ Caution | Go {direction}")
        else:
            self.status_label.config(text=f"Status: 🟢 Safe | Go {direction}")

        # -----------------------------
        # Display in Tkinter
        # -----------------------------
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)

        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo

        if self.camera_running:
            self.root.after(15, self.update_frame)

    def stop_webcam(self) -> None:
        """Stop the webcam without closing the app."""
        self.camera_running = False

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.cap = None

        # Restore the preview placeholder
        self.preview_label.config(
            image="",
            text="Camera / Video Preview"
        )
        self.preview_label.image = None
        self.status_label.config(text="Status: Webcam stopped")

    def on_close(self) -> None:
        """Release webcam and stop background voice thread before closing."""
        self.feature_logger.close()
        self.camera_running = False

        if hasattr(self, "voice") and self.voice:
            self.voice.stop()

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.root.destroy()

    def upload_video(self) -> None:
        """Choose a video file and start playback."""
        if self.camera_running:
            self.stop_webcam()

        file_path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.cap = cv2.VideoCapture(file_path)
        self.video_path = file_path
        self.frame_count = 0
        self.cached_custom_results = None
        self.cached_coco_results = None
        self.cached_depth = None
        self.cached_seg_map = None
        self.cached_walkable_mask = None
        self.navigator.reset()

        

        if not self.cap.isOpened():
            self.status_label.config(text="Status: Unable to open video")
            return

        self.camera_running = True
        self.status_label.config(text="Status: Playing video")
        self.update_frame()


def main() -> None:
    """Create the root window and launch the VisionAssist app."""
    root = tk.Tk()
    app = VisionAssistApp(root)

    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()