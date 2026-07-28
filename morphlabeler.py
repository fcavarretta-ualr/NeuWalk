"""Interactive Matplotlib SWC viewer/editor.

Version 3.0

Features
--------
- Load an SWC file as a list of root Neurite sections.
- Color sections according to section_type.
- Click a section to select it and increase its line width.
- Clicking the selected section again keeps it selected.
- Click empty plot space to deselect the section.
- Preserve the current 3D camera and zoom when selecting or editing.
- Change the selected section_type.
- Apply the selected type to the selected section and all descendants.
- Save the edited morphology with write_swc from swc.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import proj3d

try:
    from neuwalk.io.swc import TYPE_LABELS, load_swc, write_swc
except ImportError:
    from neuwalk.io.swc import TYPE_LABELS, read_swc as load_swc, write_swc


VERSION = "3.0"
TYPE_COLORS = {
    "unknown": "tab:gray",
    "soma": "black",
    "axon": "tab:orange",
    "basal_dendrite": "tab:blue",
    "apical_dendrite": "tab:red",
    "apical_oblique": "tab:green",
    "apical_secondary_dendrite": "tab:purple",
    "apical_secondary_oblique": "tab:brown",
}
SECTION_TYPES = tuple(TYPE_LABELS.values())
NORMAL_WIDTH = 1.5
SELECTED_WIDTH = 5.0
NORMAL_MARKER_SIZE = 5.0
SELECTED_MARKER_SIZE = 10.0
CLICK_TOLERANCE = 5.0
PICK_TOLERANCE = 8.0


class SWCViewer:
    def __init__(self, roots, filename):
        self.roots = list(roots)
        self.filename = Path(filename)
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes((0.05, 0.08, 0.70, 0.86), projection="3d")
        self.artist_to_section = {}
        self.section_to_artist = {}
        self.selected_artist = None
        self.press_xy = None
        self.updating_type_control = False
        self.status_text = self.fig.text(0.77, 0.06, "No section selected.", ha="left", va="bottom", wrap=True)
        self._plot_sections()
        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    @property
    def sections(self):
        return [section for root in self.roots for section in root.subtree]

    @property
    def selected_section(self):
        return None if self.selected_artist is None else self.artist_to_section[self.selected_artist]

    def _plot_sections(self):
        all_points = []
        for section in self.sections:
            points = np.asarray(section.points, dtype=float)
            if len(points) == 0:
                continue

            display_points = points
            if section.parent is not None and len(section.parent.points):
                parent_endpoint = np.asarray(section.parent.points[-1], dtype=float)
                if not np.allclose(points[0], parent_endpoint):
                    display_points = np.vstack((parent_endpoint, points))

            section_type = section.section_type or "unknown"
            artist, = self.ax.plot(
                display_points[:, 0], display_points[:, 1], display_points[:, 2],
                color=self._color(section_type), linewidth=NORMAL_WIDTH,
                marker="o" if len(display_points) == 1 else None,
                markersize=NORMAL_MARKER_SIZE,
            )
            self.artist_to_section[artist] = section
            self.section_to_artist[id(section)] = artist
            all_points.append(display_points)

        if not self.artist_to_section:
            raise ValueError("The morphology contains no section points to display.")

        self._set_equal_axes(np.vstack(all_points))
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self._update_title()
        handles = [Line2D([0], [0], color=self._color(label), linewidth=2, label=label) for label in SECTION_TYPES]
        self.ax.legend(handles=handles, loc="upper right", fontsize=8)

    def _build_controls(self):
        radio_ax = self.fig.add_axes((0.78, 0.39, 0.20, 0.50), facecolor="none")
        radio_ax.set_title("Selected section type", loc="left", fontsize=10)
        self.type_radio = RadioButtons(radio_ax, SECTION_TYPES, active=0)
        self.type_radio.on_clicked(self._change_selected_type)

        subtree_ax = self.fig.add_axes((0.78, 0.27, 0.20, 0.07))
        self.subtree_button = Button(subtree_ax, "Apply type to subtree")
        self.subtree_button.on_clicked(self._apply_type_to_subtree)

        save_ax = self.fig.add_axes((0.78, 0.17, 0.20, 0.07))
        self.save_button = Button(save_ax, "Save SWC")
        self.save_button.on_clicked(self._save)

    def _on_press(self, event):
        if event.button == MouseButton.LEFT and event.inaxes is self.ax and event.x is not None and event.y is not None:
            self.press_xy = (event.x, event.y)
        else:
            self.press_xy = None

    def _on_release(self, event):
        if event.button != MouseButton.LEFT or self.press_xy is None:
            return

        press_xy = self.press_xy
        self.press_xy = None
        if event.inaxes is not self.ax or event.x is None or event.y is None:
            return

        if np.hypot(event.x - press_xy[0], event.y - press_xy[1]) > CLICK_TOLERANCE:
            return

        self._select(self._artist_at(event.x, event.y))

    def _artist_at(self, x, y):
        closest_artist = None
        closest_distance = np.inf
        projection = self.ax.get_proj()

        for artist in self.artist_to_section:
            x3, y3, z3 = artist.get_data_3d()
            x2, y2, _ = proj3d.proj_transform(np.asarray(x3), np.asarray(y3), np.asarray(z3), projection)
            screen_points = self.ax.transData.transform(np.column_stack((x2, y2)))
            screen_points = screen_points[np.all(np.isfinite(screen_points), axis=1)]
            distance = self._polyline_distance(np.array((x, y), dtype=float), screen_points)

            if distance < closest_distance:
                closest_artist = artist
                closest_distance = distance

        return closest_artist if closest_distance <= PICK_TOLERANCE else None

    @staticmethod
    def _polyline_distance(point, points):
        if len(points) == 0:
            return np.inf
        if len(points) == 1:
            return np.linalg.norm(point - points[0])

        starts = points[:-1]
        vectors = points[1:] - starts
        lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
        parameters = np.zeros(len(vectors), dtype=float)
        valid = lengths_squared > 0.0
        parameters[valid] = np.einsum("ij,ij->i", point - starts[valid], vectors[valid]) / lengths_squared[valid]
        parameters = np.clip(parameters, 0.0, 1.0)
        nearest = starts + parameters[:, None] * vectors
        return np.min(np.linalg.norm(nearest - point, axis=1))

    def _select(self, artist):
        view = self._capture_view()

        if self.selected_artist is not None and self.selected_artist is not artist:
            self._set_artist_selected(self.selected_artist, False)

        self.selected_artist = artist
        if artist is None:
            self.status_text.set_text("No section selected.")
        else:
            self._set_artist_selected(artist, True)
            self._set_type_control(self.selected_section.section_type or "unknown")
            self._update_status()

        self._update_title()
        self._restore_view(view)
        self.fig.canvas.draw_idle()

    def _change_selected_type(self, section_type):
        if self.updating_type_control or self.selected_section is None:
            return

        view = self._capture_view()
        self.selected_section.section_type = section_type
        self._update_artist_color(self.selected_section)
        self._update_status()
        self._update_title()
        self._restore_view(view)
        self.fig.canvas.draw_idle()

    def _apply_type_to_subtree(self, _event):
        if self.selected_section is None:
            self.status_text.set_text("Select a section first.")
            self.fig.canvas.draw_idle()
            return

        view = self._capture_view()
        section_type = self.type_radio.value_selected
        subtree = self.selected_section.subtree

        for section in subtree:
            section.section_type = section_type
            self._update_artist_color(section)

        self._update_status(f"Applied {section_type} to {len(subtree)} section(s).")
        self._update_title()
        self._restore_view(view)
        self.fig.canvas.draw_idle()

    def _save(self, _event):
        from tkinter import Tk, filedialog, messagebox

        dialog = Tk()
        dialog.withdraw()
        dialog.attributes("-topmost", True)
        output = filedialog.asksaveasfilename(
            title="Save SWC morphology",
            initialdir=str(self.filename.parent),
            initialfile=f"{self.filename.stem}_edited.swc",
            defaultextension=".swc",
            filetypes=[("SWC files", "*.swc"), ("All files", "*")],
        )

        if not output:
            dialog.destroy()
            return

        try:
            write_swc(output, self.roots)
        except Exception as error:
            messagebox.showerror("Save failed", str(error), parent=dialog)
            self.status_text.set_text(f"Save failed: {error}")
        else:
            self.filename = Path(output)
            self.status_text.set_text(f"Saved: {self.filename}")
            self._update_title()

        dialog.destroy()
        self.fig.canvas.draw_idle()

    def _set_artist_selected(self, artist, selected):
        artist.set_linewidth(SELECTED_WIDTH if selected else NORMAL_WIDTH)
        artist.set_markersize(SELECTED_MARKER_SIZE if selected else NORMAL_MARKER_SIZE)

    def _update_artist_color(self, section):
        artist = self.section_to_artist.get(id(section))
        if artist is not None:
            artist.set_color(self._color(section.section_type or "unknown"))

    def _update_status(self, message=None):
        section = self.selected_section
        if section is None:
            self.status_text.set_text("No section selected.")
            return
        section_type = section.section_type or "unknown"
        self.status_text.set_text(message or f"Selected: {section_type}\nPoints: {len(section.points)}\nLength: {section.length:.3f}")

    def _update_title(self):
        section = self.selected_section
        if section is None:
            self.ax.set_title(f"{self.filename.name} — SWC Viewer {VERSION}")
            return
        section_type = section.section_type or "unknown"
        self.ax.set_title(f"{self.filename.name} — SWC Viewer {VERSION}\nSelected: {section_type} | points: {len(section.points)} | length: {section.length:.3f}")

    def _set_type_control(self, section_type):
        section_type = section_type if section_type in SECTION_TYPES else "unknown"
        if self.type_radio.value_selected == section_type:
            return
        self.updating_type_control = True
        self.type_radio.set_active(SECTION_TYPES.index(section_type))
        self.updating_type_control = False

    def _set_equal_axes(self, points):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2.0
        radius = max((maximum - minimum).max() / 2.0, 1.0)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.ax.set_box_aspect((1, 1, 1))

    def _capture_view(self):
        return {
            "elev": self.ax.elev,
            "azim": self.ax.azim,
            "roll": getattr(self.ax, "roll", 0.0),
            "xlim": self.ax.get_xlim3d(),
            "ylim": self.ax.get_ylim3d(),
            "zlim": self.ax.get_zlim3d(),
        }

    def _restore_view(self, view):
        try:
            self.ax.view_init(elev=view["elev"], azim=view["azim"], roll=view["roll"])
        except TypeError:
            self.ax.view_init(elev=view["elev"], azim=view["azim"])
        self.ax.set_xlim3d(view["xlim"])
        self.ax.set_ylim3d(view["ylim"])
        self.ax.set_zlim3d(view["zlim"])

    @staticmethod
    def _color(section_type):
        return TYPE_COLORS.get(section_type, "tab:pink")

    def show(self):
        plt.show()


def choose_file():
    from tkinter import Tk, filedialog

    dialog = Tk()
    dialog.withdraw()
    dialog.attributes("-topmost", True)
    filename = filedialog.askopenfilename(
        title="Open SWC morphology",
        filetypes=[("SWC files", "*.swc"), ("All files", "*")],
    )
    dialog.destroy()
    return filename


def main():
    parser = argparse.ArgumentParser(description="Visualize, edit, and save SWC section types.")
    parser.add_argument("filename", nargs="?", help="SWC file to display")
    args = parser.parse_args()
    filename = args.filename or choose_file()
    if filename:
        SWCViewer(load_swc(filename), filename).show()


if __name__ == "__main__":
    main()
