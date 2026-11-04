"""Desktop polygon editor for manually separating visible surface regions."""

from __future__ import annotations

from copy import deepcopy
import json
import numpy as np

from depth_anything_3.utils.regions import rasterize_regions


REGION_COLORS = np.array([[40, 190, 230], [255, 175, 50], [165, 100, 230],
                          [80, 205, 120], [235, 105, 140], [180, 180, 65]], np.uint8)


def region_rgb(data):
    h, w = data["valid_mask"].shape
    rgb = np.full((h, w, 3), 235, np.uint8)
    if "colors" in data:
        u, v = data["pixels_uv"].T
        rgb[v, u] = data["colors"]
    return rgb


class PolygonEditor:
    """Annotation state shared by the GUI and headless event tests."""

    def __init__(self, annotation, image_shape):
        self.data = deepcopy(annotation)
        self.image_shape = image_shape
        self.active, self.holes = 0, False
        self.current, self.history = [], []
        for i, region in enumerate(self.data["regions"]):
            region.setdefault("holes", [])
            for kind in ("polygons", "holes"):
                for _ in region[kind]:
                    self.history.append((i, kind))

    def payload(self):
        return {**self.data, "regions": [deepcopy(r) for r in self.data["regions"] if r["polygons"]]}

    def finish(self):
        if len(self.current) < 3:
            raise ValueError("Click at least three vertices before Enter")
        kind = "holes" if self.holes else "polygons"
        region = self.data["regions"][self.active]
        if self.holes and not region["polygons"]:
            raise ValueError("Draw an outer polygon before adding a hole")
        region[kind].append(deepcopy(self.current))
        try:
            rasterize_regions(self.payload(), self.image_shape)
        except ValueError:
            region[kind].pop()
            raise
        self.history.append((self.active, kind))
        self.current.clear()

    def undo(self):
        if self.current:
            self.current.pop()
        elif self.history:
            i, kind = self.history.pop()
            self.data["regions"][i][kind].pop()


def annotate_regions(data, output, *, initial=None, image_path=None, resize_only=False, matrix=None):
    """Open a GUI only when explicitly invoked; return the saved annotation."""
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    rgb = region_rgb(data)
    h, w = data["valid_mask"].shape
    if image_path is not None:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
    ih, iw = rgb.shape[:2]
    if matrix is None and resize_only:
        matrix = np.diag([w/iw, h/ih, 1.])
    if matrix is None and initial is not None:
        matrix = initial.get("original_to_processed")
    if matrix is None:
        if (iw, ih) == (w, h):
            matrix = np.eye(3)
        else:
            raise ValueError("Annotation image size differs; use --annotation-resize-only or --annotation-to-processed")
    if initial is None:
        initial = dict(image_size=[iw, ih], original_to_processed=np.asarray(matrix).tolist(), regions=[
            dict(id=1, name="left", polygons=[], holes=[]),
            dict(id=2, name="right", polygons=[], holes=[]),
        ])
    else:
        if initial.get("image_size") != [iw, ih]:
            raise ValueError("Existing annotation image_size does not match the displayed image")
        initial = {**initial, "original_to_processed": np.asarray(matrix).tolist()}
        rasterize_regions(initial, (h, w))
    editor = PolygonEditor(initial, (h, w))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb)
    artists, saved = [], []

    def redraw(message=""):
        for artist in artists:
            artist.remove()
        artists.clear()
        for i, region in enumerate(editor.data["regions"]):
            color = REGION_COLORS[i % len(REGION_COLORS)]/255
            for kind in ("polygons", "holes"):
                for vertices in region[kind]:
                    patch = Polygon(vertices, facecolor=color if kind == "polygons" else "none",
                                    edgecolor=color, alpha=.3 if kind == "polygons" else 1,
                                    hatch="///" if kind == "holes" else None)
                    ax.add_patch(patch)
                    artists.append(patch)
        if editor.current:
            p = np.asarray(editor.current)
            artists.extend(ax.plot(p[:, 0], p[:, 1], "o-", color=REGION_COLORS[editor.active % len(REGION_COLORS)]/255))
        region = editor.data["regions"][editor.active]
        mode = "hole" if editor.holes else "outer polygon"
        names = " | ".join(f"{i+1}: {r['name']}" for i, r in enumerate(editor.data["regions"]))
        ax.set_title(f"{names} | Active: {region['name']} ({mode})\n"
                     "Click vertices | Enter: close | H: hole | Backspace: undo | S: save and fit | Esc: cancel\n"
                     "Wheel: zoom | Toolbar: zoom/pan | " + message, fontsize=10)
        fig.canvas.draw_idle()

    def click(event):
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.inaxes is ax and event.button == 1 and event.xdata is not None:
            if 0 <= event.xdata <= iw-1 and 0 <= event.ydata <= ih-1:
                editor.current.append([float(event.xdata), float(event.ydata)])
                redraw()

    def key(event):
        try:
            if event.key == "enter":
                editor.finish()
            elif event.key == "backspace":
                editor.undo()
            elif event.key in ("h", "H"):
                if editor.current:
                    raise ValueError("Finish or undo the current polygon first")
                editor.holes = not editor.holes
            elif event.key in [str(i+1) for i in range(min(9, len(editor.data["regions"])) )]:
                if editor.current:
                    raise ValueError("Finish or undo the current polygon before changing region")
                editor.active = int(event.key)-1
                editor.holes = False
            elif event.key in ("s", "S"):
                if editor.current:
                    raise ValueError("Press Enter to close the current polygon before saving")
                payload = editor.payload()
                rasterize_regions(payload, (h, w))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                saved.append(payload)
                plt.close(fig)
                return
            elif event.key == "escape":
                plt.close(fig)
                return
            else:
                return
            redraw()
        except ValueError as exc:
            redraw(str(exc))

    def scroll(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        scale = .8 if event.button == "up" else 1.25
        for getter, setter, center in ((ax.get_xlim, ax.set_xlim, event.xdata),
                                       (ax.get_ylim, ax.set_ylim, event.ydata)):
            setter([center+(edge-center)*scale for edge in getter()])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", key)
    fig.canvas.mpl_connect("scroll_event", scroll)
    # Avoid default Matplotlib S/Backspace key bindings consuming editor actions.
    manager = fig.canvas.manager
    if getattr(manager, "key_press_handler_id", None) is not None:
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)
    redraw()
    plt.show()
    if not saved:
        raise ValueError("Region annotation cancelled or GUI unavailable; use a desktop backend or --regions JSON")
    return saved[0]


def save_region_preview(output, data, processed):
    from PIL import Image
    from scipy.ndimage import binary_erosion

    labels = data["region_mask"]
    rgb = region_rgb({**data, "valid_mask": np.ones(labels.shape, bool)})
    overlay = rgb.copy()
    for i, region in enumerate(processed["regions"]):
        selected = labels == region["id"]
        color = REGION_COLORS[i % len(REGION_COLORS)]
        overlay[selected] = np.rint(.72*rgb[selected]+.28*color).astype(np.uint8)
        overlay[selected & ~binary_erosion(selected)] = color
    Image.fromarray(labels.astype(np.uint16)).save(output / "region_mask.png")
    Image.fromarray(overlay).save(output / "region_overlay.png")
