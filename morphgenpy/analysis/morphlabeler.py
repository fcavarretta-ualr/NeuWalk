from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

    def section_points(self, section_id: int) -> np.ndarray:
        section = self.section_by_id[section_id]
        return np.vstack([self.coords[node] for node in section.nodes])

    def section_polyline(self, section_id: int, projection: str = "xy") -> tuple[np.ndarray, np.ndarray]:
        points = self.section_points(section_id)

        if projection == "xy":
            return points[:, 0], points[:, 1]
        if projection == "xz":
            return points[:, 0], points[:, 2]
        if projection == "yz":
            return points[:, 1], points[:, 2]

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

    def save_outputs(self, swc_path: str | Path):
        swc_path = Path(swc_path)

        if swc_path.suffix.lower() != ".swc":
            swc_path = swc_path.with_suffix(".swc")

        swc_path.parent.mkdir(parents=True, exist_ok=True)
        node_df = self.node_labels_dataframe()

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write("# Custom manually labeled SWC\n")

            if self.path is not None:
                f.write(f"# Source file: {self.path}\n")

            for label, code in sorted(LABEL_CODE.items(), key=lambda item: item[1]):
                f.write(f"#   {code}: {label}\n")

            f.write("# Columns: id custom_type_code x y z radius parent\n")

            for row in node_df.itertuples(index=False):
                f.write(
                    f"{int(row.id)} {int(row.custom_type_code)} "
                    f"{row.x:.6f} {row.y:.6f} {row.z:.6f} "
                    f"{row.r:.6f} {int(row.parent)}\n"
                )

        return swc_path

    def load_label_session(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        labels = session.get("section_labels", {})

        # Backward compatibility with older sessions that separated
        # apical_bifurcation / apical_bifurcation and their obliques.
        merge_old_labels = {
            "apical_bifurcation": "apical_bifurcation",
            "apical_bifurcation_oblique": "apical_bifurcation_oblique",
        }

        for sid_str, label in labels.items():
            sid = int(sid_str)
            label = merge_old_labels.get(label, label)

            if sid in self.section_by_id and label in LABEL_CODE:
                self.section_labels[sid] = label
