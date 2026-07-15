#!/usr/bin/env python3
"""Interactive manual SWC dendrite labeler."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from morphgenpy.analysis.morphlabeler import LABEL_COLORS, LABELS_IN_ORDER, SWCModel


class LabelerApp(tk.Tk):
    def __init__(self, swc_path: str | None = None):
        super().__init__()

        self.title("Manual SWC Dendrite Labeler")
        self.geometry("1250x850")

        self.model: SWCModel | None = None
        self.current_file: Path | None = None
        self.selected_section_id: int | None = None
        self.label_var = tk.StringVar(value="unknown")
        self.status_var = tk.StringVar(value="Open an SWC file to begin.")
        self.inherit_unknown_var = tk.BooleanVar(value=True)
        self.line_by_section: dict[int, object] = {}
        self._mouse_press_xy: tuple[float, float] | None = None

        self._build_menu()
        self._build_layout()

        if swc_path:
            self.open_swc(swc_path)

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open SWC...", command=self.open_dialog)
        file_menu.add_command(label="Load label session...", command=self.load_session_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save SWC...", command=self.save_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_layout(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = ttk.Frame(outer, padding=10)
        panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.mouse_init()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        ttk.Label(
            panel,
            text="3D navigation\nLeft-drag: rotate\nRight-drag or scroll: zoom",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 10))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text="Selected section").pack(anchor="w")
        self.selected_label = ttk.Label(panel, text="None", width=38)
        self.selected_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(panel, text="Set label").pack(anchor="w")
        ttk.Combobox(
            panel,
            textvariable=self.label_var,
            values=LABELS_IN_ORDER,
            state="readonly",
            width=35,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(panel, text="Apply to selected section", command=self.apply_to_selected).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Apply to downstream subtree", command=self.apply_to_subtree).pack(fill=tk.X, pady=3)

        ttk.Checkbutton(
            panel,
            text="If selected is unknown, inherit label to descendants",
            variable=self.inherit_unknown_var,
        ).pack(anchor="w", pady=(4, 2))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text="Quick label buttons").pack(anchor="w")

        for label in LABELS_IN_ORDER:
            bg = LABEL_COLORS.get(label, "#cccccc")
            fg = "black" if label.endswith("_oblique") else "white"
            tk.Button(
                panel,
                text=label.replace("_", " "),
                bg=bg,
                fg=fg,
                command=lambda lab=label: self.quick_apply(lab),
                anchor="w",
                relief=tk.RAISED,
            ).pack(fill=tk.X, pady=1)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Button(panel, text="Save SWC", command=self.save_dialog).pack(fill=tk.X, pady=3)
        ttk.Label(panel, textvariable=self.status_var, wraplength=270, justify=tk.LEFT).pack(anchor="w", pady=(12, 0))

    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open SWC file",
            filetypes=[("SWC files", "*.swc"), ("All files", "*.*")],
        )
        if path:
            self.open_swc(path)

    def open_swc(self, path: str | Path):
        try:
            self.model = SWCModel(path)
        except Exception as error:
            messagebox.showerror("Could not open SWC", str(error))
            return

        self.current_file = Path(path)
        self.selected_section_id = None
        self.status_var.set(f"Loaded {self.current_file.name}. Rotate the morphology and click a section.")
        self.redraw()

    def load_session_dialog(self):
        if self.model is None:
            messagebox.showinfo("No SWC loaded", "Open an SWC file before loading a label session.")
            return

        path = filedialog.askopenfilename(
            title="Load label session JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.model.load_label_session(path)
        except Exception as error:
            messagebox.showerror("Could not load session", str(error))
            return

        self.status_var.set(f"Loaded label session: {Path(path).name}")
        self.redraw(preserve_view=True)

    def save_dialog(self):
        if self.model is None:
            messagebox.showinfo("No SWC loaded", "Open an SWC file before saving.")
            return

        default = self.current_file.stem + "_curated.swc" if self.current_file else "curated_labels.swc"
        path = filedialog.asksaveasfilename(
            title="Save labeled SWC",
            initialfile=default,
            defaultextension=".swc",
            filetypes=[("SWC files", "*.swc")],
            confirmoverwrite=True,
        )
        if not path:
            return

        try:
            swc_path = self.model.save_outputs(path)
        except Exception as error:
            messagebox.showerror("Could not save SWC", str(error))
            return

        self.status_var.set(f"Saved {swc_path.name}.")
        messagebox.showinfo("Saved", str(swc_path))

    def redraw(self, preserve_view: bool = False):
        had_plot = bool(self.line_by_section)
        view = None

        if preserve_view and had_plot:
            view = (
                self.ax.elev,
                self.ax.azim,
                self.ax.get_xlim3d(),
                self.ax.get_ylim3d(),
                self.ax.get_zlim3d(),
            )

        self.ax.clear()
        self.ax.mouse_init()
        self.line_by_section = {}

        if self.model is None:
            self.ax.set_title("Open an SWC file")
            self.canvas.draw_idle()
            return

        all_points = []

        for section in self.model.sections:
            sid = section.section_id
            points = self.model.section_points(sid)
            all_points.append(points)

            label = self.model.section_label(sid)
            color = LABEL_COLORS.get(label, "#7f7f7f")
            width = 3.2 if sid == self.selected_section_id else 1.3
            alpha = 1.0 if sid == self.selected_section_id else 0.95

            line, = self.ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                lw=width,
                alpha=alpha,
                solid_capstyle="round",
                picker=6,
            )
            line.set_gid(sid)
            self.line_by_section[sid] = line

        if view is not None:
            elev, azim, xlim, ylim, zlim = view
            self.ax.view_init(elev=elev, azim=azim)
            self.ax.set_xlim3d(xlim)
            self.ax.set_ylim3d(ylim)
            self.ax.set_zlim3d(zlim)
        elif all_points:
            points = np.vstack(all_points)
            minima = points.min(axis=0)
            maxima = points.max(axis=0)
            spans = maxima - minima
            padding = np.maximum(0.05 * spans, 5.0)
            self.ax.set_xlim3d(minima[0] - padding[0], maxima[0] + padding[0])
            self.ax.set_ylim3d(minima[1] - padding[1], maxima[1] + padding[1])
            self.ax.set_zlim3d(minima[2] - padding[2], maxima[2] + padding[2])
            self.ax.set_box_aspect(np.maximum(spans, 1.0))

        self.ax.set_xlabel("x (µm)")
        self.ax.set_ylabel("y (µm)")
        self.ax.set_zlabel("z (µm)")
        title = self.current_file.name if self.current_file else "SWC morphology"
        self.ax.set_title(f"{title} — interactive 3D view")

        present = sorted(
            set(self.model.section_labels.values()),
            key=lambda label: LABELS_IN_ORDER.index(label) if label in LABELS_IN_ORDER else 999,
        )
        handles = [
            Line2D([0], [0], color=LABEL_COLORS.get(label, "#7f7f7f"), lw=2, label=label.replace("_", " "))
            for label in present
        ]
        if handles:
            self.ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

        self.canvas.draw_idle()
        self._update_selected_text()

    def _on_mouse_press(self, event):
        if event.button == 1:
            self._mouse_press_xy = (event.x, event.y)

    def _on_mouse_release(self, event):
        if event.button != 1 or self._mouse_press_xy is None:
            return

        x, y = self._mouse_press_xy
        self._mouse_press_xy = None

        if np.hypot(event.x - x, event.y - y) > 5:
            return

        if any(line.contains(event)[0] for line in self.line_by_section.values()):
            return

        if self.selected_section_id is not None:
            self.selected_section_id = None
            self.status_var.set("Deselected section.")
            self.redraw(preserve_view=True)

    def _on_pick(self, event):
        sid = event.artist.get_gid()

        if sid is None:
            return

        self.selected_section_id = int(sid)

        if self.model is not None:
            self.label_var.set(self.model.section_label(self.selected_section_id))

        self.status_var.set(f"Selected section {self.selected_section_id}.")
        self._update_selected_text()
        self.redraw(preserve_view=True)

    def _update_selected_text(self):
        if self.model is None or self.selected_section_id is None:
            self.selected_label.config(text="None")
            return

        section = self.model.section_by_id[self.selected_section_id]
        label = self.model.section_label(self.selected_section_id)
        self.selected_label.config(
            text=(
                f"Section {section.section_id}\n"
                f"Current label: {label}\n"
                f"Nodes: {section.start_node} → {section.end_node}\n"
                f"n = {len(section.nodes)}, original SWC type = {section.original_type}"
            )
        )

    def apply_to_selected(self):
        if self.model is None or self.selected_section_id is None:
            messagebox.showinfo("No section selected", "Click a section first.")
            return

        label = self.label_var.get()
        old_label = self.model.section_label(self.selected_section_id)

        if old_label == "unknown" and self.inherit_unknown_var.get():
            count = len(self.model.downstream_sections(self.selected_section_id))
            self.model.set_subtree_label(self.selected_section_id, label)
            self.status_var.set(
                f"Selected section was unknown: set section {self.selected_section_id} "
                f"and {count - 1} downstream sections to {label}."
            )
        else:
            self.model.set_section_label(self.selected_section_id, label)
            self.status_var.set(f"Set section {self.selected_section_id} to {label}.")

        self.redraw(preserve_view=True)

    def apply_to_subtree(self):
        if self.model is None or self.selected_section_id is None:
            messagebox.showinfo("No section selected", "Click a section first.")
            return

        label = self.label_var.get()
        count = len(self.model.downstream_sections(self.selected_section_id))
        self.model.set_subtree_label(self.selected_section_id, label)
        self.status_var.set(
            f"Set section {self.selected_section_id} and {count - 1} downstream sections to {label}."
        )
        self.redraw(preserve_view=True)

    def quick_apply(self, label: str):
        self.label_var.set(label)
        self.apply_to_selected()


def main():
    parser = argparse.ArgumentParser(description="Interactive manual SWC dendrite labeler.")
    parser.add_argument("swc", nargs="?", help="Optional SWC file to open.")
    args = parser.parse_args()

    app = LabelerApp(args.swc)
    app.mainloop()


if __name__ == "__main__":
    main()
