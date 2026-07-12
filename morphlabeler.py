#!/usr/bin/env python3
"""
manual_swc_labeler.py

Interactive manual SWC dendrite labeler.

Usage:
    python manual_swc_labeler.py path/to/morphology.swc

Requirements:
    pip install numpy pandas matplotlib

Features:
    - Load an SWC file.
    - Split morphology into clickable unbranched sections.
    - Click a section.
    - Assign a dendritic type from a dropdown or quick buttons.
    - Colors update immediately.
    - Apply label to selected section or downstream subtree.
    - Save:
        *_custom_label_types.swc
        *_node_labels.csv
        *_section_labels.csv
        *_label_session.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D


LABEL_CODE = {
    "unknown": 0,
    "soma": 1,
    "axon": 2,
    "basal_dendrite": 3,
    "primary_apical_trunk": 4,
    "primary_apical_trunk_oblique": 5,
    "apical_bifurcation": 6,
    "apical_bifurcation_oblique": 7,
}

LABELS_IN_ORDER = [
    "soma",
    "axon",
    "basal_dendrite",
    "primary_apical_trunk",
    "primary_apical_trunk_oblique",
    "apical_bifurcation",
    "apical_bifurcation_oblique",
    "unknown",
]

LABEL_COLORS = {
    "unknown": "#7f7f7f",
    "soma": "#000000",
    "axon": "#8c8c8c",
    "basal_dendrite": "#1f77b4",
    "primary_apical_trunk": "#d62728",
    "primary_apical_trunk_oblique": "#ff9896",
    "apical_bifurcation": "#2ca02c",
    "apical_bifurcation_oblique": "#98df8a",
}

DEFAULT_LABEL_BY_SWC_TYPE = {
    0: "unknown",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "primary_apical_trunk",
    5: "primary_apical_trunk_oblique",
    6: "apical_bifurcation",
    7: "apical_bifurcation_oblique",
}


@dataclass
class Section:
    section_id: int
    nodes: list[int]
    original_type: int
    parent_section_id: int | None = None

    @property
    def start_node(self) -> int:
        return self.nodes[0]

    @property
    def end_node(self) -> int:
        return self.nodes[-1]


class SWCModel:
    def __init__(self, path: str | Path | None = None):
        self.path: Path | None = None
        self.comments: list[str] = []
        self.df: pd.DataFrame | None = None
        self.children: dict[int, list[int]] = {}
        self.parent: dict[int, int] = {}
        self.types: dict[int, int] = {}
        self.coords: dict[int, np.ndarray] = {}
        self.sections: list[Section] = []
        self.section_by_id: dict[int, Section] = {}
        self.node_to_section: dict[int, int] = {}
        self.section_children: dict[int, list[int]] = {}
        self.section_labels: dict[int, str] = {}

        if path is not None:
            self.load(path)

    def load(self, path: str | Path):
        self.path = Path(path)
        self.comments, self.df = self._read_swc(self.path)
        self._build_graph()
        self._build_sections()
        self._initialize_labels()

    @staticmethod
    def _read_swc(path: Path) -> tuple[list[str], pd.DataFrame]:
        comments = []
        rows = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue

                if s.startswith("#"):
                    comments.append(line.rstrip("\n"))
                    continue

                parts = s.split()
                if len(parts) < 7:
                    continue

                rows.append([
                    int(float(parts[0])),
                    int(float(parts[1])),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    int(float(parts[6])),
                ])

        if not rows:
            raise ValueError(f"No SWC rows found in {path}")

        df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
        return comments, df

    def _build_graph(self):
        assert self.df is not None

        ids = set(self.df["id"].astype(int))
        children = defaultdict(list)

        self.parent = {}
        self.types = {}
        self.coords = {}

        for row in self.df.itertuples(index=False):
            nid = int(row.id)
            pid = int(row.parent)

            self.parent[nid] = pid
            self.types[nid] = int(row.type)
            self.coords[nid] = np.array([row.x, row.y, row.z], dtype=float)

            if pid in ids:
                children[pid].append(nid)

        self.children = {n: sorted(ch) for n, ch in children.items()}

    def _is_section_start(self, nid: int) -> bool:
        pid = self.parent.get(nid, -1)

        if pid not in self.types:
            return True

        if self.types[nid] != self.types[pid]:
            return True

        if len(self.children.get(pid, [])) != 1:
            return True

        return False

    def _build_sections(self):
        self.sections = []
        self.section_by_id = {}
        self.node_to_section = {}

        ids = set(self.types.keys())
        starts = sorted([nid for nid in ids if self._is_section_start(nid)])

        visited = set()
        sid = 1

        for start in starts:
            if start in visited:
                continue

            nodes = [start]
            visited.add(start)
            current = start

            while True:
                ch = self.children.get(current, [])

                if len(ch) != 1:
                    break

                nxt = ch[0]

                if self.types[nxt] != self.types[current]:
                    break

                if self._is_section_start(nxt):
                    break

                nodes.append(nxt)
                visited.add(nxt)
                current = nxt

            section = Section(section_id=sid, nodes=nodes, original_type=self.types[start])
            self.sections.append(section)
            self.section_by_id[sid] = section

            for n in nodes:
                self.node_to_section[n] = sid

            sid += 1

        # Defensive fallback for any missed nodes.
        for nid in sorted(ids):
            if nid not in self.node_to_section:
                section = Section(section_id=sid, nodes=[nid], original_type=self.types[nid])
                self.sections.append(section)
                self.section_by_id[sid] = section
                self.node_to_section[nid] = sid
                sid += 1

        # Section-level topology.
        self.section_children = {s.section_id: [] for s in self.sections}

        for section in self.sections:
            start = section.start_node
            pid = self.parent.get(start, -1)

            if pid in self.node_to_section:
                parent_sid = self.node_to_section[pid]

                if parent_sid != section.section_id:
                    section.parent_section_id = parent_sid
                    self.section_children.setdefault(parent_sid, []).append(section.section_id)

    def _initialize_labels(self):
        self.section_labels = {}

        for section in self.sections:
            self.section_labels[section.section_id] = DEFAULT_LABEL_BY_SWC_TYPE.get(
                section.original_type,
                "unknown",
            )

    def set_section_label(self, section_id: int, label: str):
        if label not in LABEL_CODE:
            raise ValueError(f"Unknown label: {label}")

        if section_id not in self.section_by_id:
            raise KeyError(f"Unknown section id: {section_id}")

        self.section_labels[section_id] = label

    def set_subtree_label(self, section_id: int, label: str):
        if label not in LABEL_CODE:
            raise ValueError(f"Unknown label: {label}")

        for sid in self.downstream_sections(section_id):
            self.section_labels[sid] = label

    def downstream_sections(self, section_id: int) -> list[int]:
        if section_id not in self.section_by_id:
            return []

        out = []
        stack = [section_id]

        while stack:
            sid = stack.pop()
            out.append(sid)
            stack.extend(self.section_children.get(sid, []))

        return out

    def section_polyline(self, section_id: int, projection: str = "xy") -> tuple[np.ndarray, np.ndarray]:
        section = self.section_by_id[section_id]
        pts = np.vstack([self.coords[n] for n in section.nodes])

        if projection == "xy":
            return pts[:, 0], pts[:, 1]

        if projection == "xz":
            return pts[:, 0], pts[:, 2]

        if projection == "yz":
            return pts[:, 1], pts[:, 2]

        raise ValueError("projection must be xy, xz, or yz")

    def section_label(self, section_id: int) -> str:
        return self.section_labels.get(section_id, "unknown")

    def node_labels_dataframe(self) -> pd.DataFrame:
        assert self.df is not None

        rows = []

        for row in self.df.itertuples(index=False):
            nid = int(row.id)
            sid = self.node_to_section[nid]
            label = self.section_labels.get(sid, "unknown")

            rows.append({
                "id": nid,
                "original_type": int(row.type),
                "x": float(row.x),
                "y": float(row.y),
                "z": float(row.z),
                "r": float(row.r),
                "parent": int(row.parent),
                "section_id": int(sid),
                "label": label,
                "custom_type_code": int(LABEL_CODE[label]),
            })

        return pd.DataFrame(rows)

    def section_summary_dataframe(self) -> pd.DataFrame:
        rows = []

        for section in self.sections:
            length = 0.0

            for n0, n1 in zip(section.nodes[:-1], section.nodes[1:]):
                length += float(np.linalg.norm(self.coords[n1] - self.coords[n0]))

            rows.append({
                "section_id": section.section_id,
                "parent_section_id": section.parent_section_id,
                "start_node": section.start_node,
                "end_node": section.end_node,
                "n_nodes": len(section.nodes),
                "original_type": section.original_type,
                "label": self.section_labels.get(section.section_id, "unknown"),
                "cable_length_within_section": length,
            })

        return pd.DataFrame(rows)

    def save_outputs(self, prefix: str | Path):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        node_df = self.node_labels_dataframe()
        section_df = self.section_summary_dataframe()

        node_csv = prefix.with_name(prefix.name + "_node_labels.csv")
        section_csv = prefix.with_name(prefix.name + "_section_labels.csv")
        swc_path = prefix.with_name(prefix.name + "_custom_label_types.swc")
        session_json = prefix.with_name(prefix.name + "_label_session.json")

        node_df.to_csv(node_csv, index=False)
        section_df.to_csv(section_csv, index=False)

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write("# Custom manually labeled SWC\n")

            if self.path is not None:
                f.write(f"# Source file: {self.path}\n")

            f.write("# Original SWC type is preserved in *_node_labels.csv\n")

            for label, code in sorted(LABEL_CODE.items(), key=lambda kv: kv[1]):
                f.write(f"#   {code}: {label}\n")

            f.write("# Columns: id custom_type_code x y z radius parent\n")

            for row in node_df.itertuples(index=False):
                f.write(
                    f"{int(row.id)} {int(row.custom_type_code)} "
                    f"{row.x:.6f} {row.y:.6f} {row.z:.6f} {row.r:.6f} {int(row.parent)}\n"
                )

        session = {
            "source_swc": str(self.path) if self.path is not None else None,
            "label_code": LABEL_CODE,
            "section_labels": {str(k): v for k, v in self.section_labels.items()},
        }

        with open(session_json, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        return {
            "swc": swc_path,
            "node_csv": node_csv,
            "section_csv": section_csv,
            "session_json": session_json,
        }

    def load_label_session(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        labels = session.get("section_labels", {})

        # Backward compatibility with older sessions that separated
        # apical_bifurcation / apical_bifurcation and their obliques.
        merge_old_labels = {
            "apical_bifurcation": "apical_bifurcation",
            "apical_bifurcation": "apical_bifurcation",
            "apical_bifurcation_oblique": "apical_bifurcation_oblique",
            "apical_bifurcation_oblique": "apical_bifurcation_oblique",
        }

        for sid_str, label in labels.items():
            sid = int(sid_str)
            label = merge_old_labels.get(label, label)

            if sid in self.section_by_id and label in LABEL_CODE:
                self.section_labels[sid] = label


class LabelerApp(tk.Tk):
    def __init__(self, swc_path: str | None = None):
        super().__init__()

        self.title("Manual SWC Dendrite Labeler")
        self.geometry("1250x850")

        self.model: SWCModel | None = None
        self.current_file: Path | None = None
        self.selected_section_id: int | None = None
        self.projection_var = tk.StringVar(value="xy")
        self.label_var = tk.StringVar(value="apical_unclassified")
        self.status_var = tk.StringVar(value="Open an SWC file to begin.")
        self.inherit_unknown_var = tk.BooleanVar(value=True)
        self.line_by_section: dict[int, Line2D] = {}

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
        file_menu.add_command(label="Save outputs...", command=self.save_dialog)
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
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        ttk.Label(panel, text="Projection").pack(anchor="w")

        projection_row = ttk.Frame(panel)
        projection_row.pack(anchor="w", pady=(0, 10))

        for p in ["xy", "xz", "yz"]:
            ttk.Radiobutton(
                projection_row,
                text=p.upper(),
                value=p,
                variable=self.projection_var,
                command=self.redraw,
            ).pack(side=tk.LEFT)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        ttk.Label(panel, text="Selected section").pack(anchor="w")
        self.selected_label = ttk.Label(panel, text="None", width=38)
        self.selected_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(panel, text="Set label").pack(anchor="w")

        label_box = ttk.Combobox(
            panel,
            textvariable=self.label_var,
            values=LABELS_IN_ORDER,
            state="readonly",
            width=35,
        )
        label_box.pack(anchor="w", pady=(0, 8))

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

        ttk.Button(panel, text="Save outputs", command=self.save_dialog).pack(fill=tk.X, pady=3)

        ttk.Label(
            panel,
            textvariable=self.status_var,
            wraplength=270,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

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
        except Exception as e:
            messagebox.showerror("Could not open SWC", str(e))
            return

        self.current_file = Path(path)
        self.selected_section_id = None
        self.status_var.set(f"Loaded {self.current_file.name}. Click a section to select it.")
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
        except Exception as e:
            messagebox.showerror("Could not load session", str(e))
            return

        self.status_var.set(f"Loaded label session: {Path(path).name}")
        self.redraw()

    def save_dialog(self):
        if self.model is None:
            messagebox.showinfo("No SWC loaded", "Open an SWC file before saving.")
            return

        default = self.current_file.stem + "_curated" if self.current_file else "curated_labels"

        path = filedialog.asksaveasfilename(
            title="Choose output prefix",
            initialfile=default,
            defaultextension="",
            filetypes=[("Output prefix", "*.*")],
        )

        if not path:
            return

        try:
            outputs = self.model.save_outputs(path)
        except Exception as e:
            messagebox.showerror("Could not save outputs", str(e))
            return

        msg = "\n".join(f"{k}: {v}" for k, v in outputs.items())
        self.status_var.set("Saved outputs.")
        messagebox.showinfo("Saved", msg)

    def redraw(self, preserve_view: bool = False):
        # Preserve current zoom/pan limits when selecting or relabeling.
        # This prevents the view from jumping back to the full morphology.
        old_xlim = self.ax.get_xlim()
        old_ylim = self.ax.get_ylim()
        had_existing_plot = bool(self.line_by_section)

        self.ax.clear()
        self.line_by_section = {}

        if self.model is None:
            self.ax.set_title("Open an SWC file")
            self.canvas.draw_idle()
            return

        projection = self.projection_var.get()
        xs_all = []
        ys_all = []

        for section in self.model.sections:
            sid = section.section_id
            x, y = self.model.section_polyline(sid, projection)
            xs_all.extend(x)
            ys_all.extend(y)

            label = self.model.section_label(sid)
            color = LABEL_COLORS.get(label, "#7f7f7f")

            lw = 3.2 if sid == self.selected_section_id else 1.3
            alpha = 1.0 if sid == self.selected_section_id else 0.95

            line, = self.ax.plot(
                x,
                y,
                color=color,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
                picker=6,
            )

            line.set_gid(sid)
            self.line_by_section[sid] = line

        if preserve_view and had_existing_plot:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
        elif xs_all and ys_all:
            xpad = max((max(xs_all) - min(xs_all)) * 0.05, 5.0)
            ypad = max((max(ys_all) - min(ys_all)) * 0.05, 5.0)

            self.ax.set_xlim(min(xs_all) - xpad, max(xs_all) + xpad)
            self.ax.set_ylim(min(ys_all) - ypad, max(ys_all) + ypad)

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel(projection[0] + " (µm)")
        self.ax.set_ylabel(projection[1] + " (µm)")

        title = self.current_file.name if self.current_file else "SWC morphology"
        self.ax.set_title(f"{title} — {projection.upper()} projection")

        present = sorted(
            set(self.model.section_labels.values()),
            key=lambda x: LABELS_IN_ORDER.index(x) if x in LABELS_IN_ORDER else 999,
        )

        handles = [
            Line2D(
                [0],
                [0],
                color=LABEL_COLORS.get(label, "#7f7f7f"),
                lw=2,
                label=label.replace("_", " "),
            )
            for label in present
        ]

        if handles:
            self.ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

        self.canvas.draw_idle()
        self._update_selected_text()

    def _on_pick(self, event):
        artist = event.artist
        sid = artist.get_gid()

        if sid is None:
            return

        self.selected_section_id = int(sid)

        if self.model is not None:
            self.label_var.set(self.model.section_label(self.selected_section_id))

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

        # If the selected section is unknown, optionally propagate the new label
        # to all downstream descendants. This is useful when a whole unlabeled
        # branch should inherit the selected dendritic identity.
        if old_label == "unknown" and self.inherit_unknown_var.get():
            n = len(self.model.downstream_sections(self.selected_section_id))
            self.model.set_subtree_label(self.selected_section_id, label)
            self.status_var.set(
                f"Selected section was unknown: set section {self.selected_section_id} "
                f"and {n - 1} downstream sections to {label}."
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
        n = len(self.model.downstream_sections(self.selected_section_id))
        self.model.set_subtree_label(self.selected_section_id, label)
        self.status_var.set(f"Set section {self.selected_section_id} and {n - 1} downstream sections to {label}.")
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
