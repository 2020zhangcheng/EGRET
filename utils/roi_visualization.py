"""Rendering helpers for ``tools/visualize_roi_points.py``.

The module is deliberately *read-only* with respect to the model: it receives
numpy arrays (point clouds, boxes, scores) and only builds pictures.  Nothing
here may resample points, recompute scores or run the network.

Conventions used everywhere in this project
-------------------------------------------
* A box is ``[cx, cy, cz, dx, dy, dz]``: the size triple holds the *extents
  along the x / y / z axes*, exactly like ``center_label`` / ``size_gts`` in the
  dataset and ``box_cxcyczwhd_to_xyzxyz`` in ``models/losses/set_criterion.py``
  (the third value is the height along ``z``).
* ``heading`` (radians) rotates the box around the ``z`` axis.  The PVGround
  checkpoints of this repository are trained with ``heading=False``, so it is
  ``0`` (axis aligned), but the rotated case is handled by the same code path.
* Every point cloud and every box is expressed in the raw ScanNet
  axis-aligned coordinate system of ``end_points['coords']`` /
  ``end_points['seed_xyz']`` (no centring, no scaling).

Rendering backends
------------------
``open3d.visualization.rendering.OffscreenRenderer`` is used when available
(headless EGL is enough, no GUI window is opened).  If the offscreen renderer
cannot be created, the panel is drawn with matplotlib's 3D projection instead,
so the script still produces the four-panel figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import cm, colors as mcolors         # noqa: E402

try:                                                 # pragma: no cover
    import open3d as o3d
except Exception:                                    # pragma: no cover
    o3d = None

RGB = Tuple[float, float, float]

# corner order: (x-,y-,z-), (x+,y-,z-), (x+,y+,z-), (x-,y+,z-), then the same
# four at z+; matches the template of ``prepare_data.boxes_to_corners_3d``.
_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],          # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],          # top face
    [0, 4], [1, 5], [2, 6], [3, 7],          # vertical edges
], dtype=np.int64)


# --------------------------------------------------------------------------- #
# box geometry
# --------------------------------------------------------------------------- #
def rotation_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix around the z axis (radians)."""
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def box_corners(center: Sequence[float], size: Sequence[float],
                heading: float = 0.0) -> np.ndarray:
    """``(8, 3)`` corners of a (possibly rotated) box.

    Args:
        center: ``[cx, cy, cz]``
        size: ``[dx, dy, dz]`` extents along x / y / z (metres)
        heading: rotation around z, radians (counter clockwise)

    Returns:
        ``(8, 3)`` float64 corners in the same coordinate system as the input.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    local = _CORNER_SIGNS * (size[None, :] * 0.5)
    if abs(float(heading)) > 1e-12:
        local = local @ rotation_z(heading).T
    return local + center[None, :]


def expand_box(center: Sequence[float], size: Sequence[float],
               delta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Grow a box by ``delta`` metres **in every direction**.

    ``delta=1`` therefore means ``dx + 2.0`` etc., while the centre and the
    heading are kept unchanged.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3).copy()
    size = np.asarray(size, dtype=np.float64).reshape(3) + 2.0 * float(delta)
    return center, size


def box_outline(center, size, heading: float = 0.0) -> np.ndarray:
    """``(12, 2, 3)`` line segments of the box (for matplotlib)."""
    corners = box_corners(center, size, heading)
    return corners[_BOX_EDGES]


# --------------------------------------------------------------------------- #
# score -> colour
# --------------------------------------------------------------------------- #
@dataclass
class ScoreColorizer:
    """Monotone ``score -> colour`` map shared by all panels of one sample.

    The very same instance is reused for the 1024 seeds and for the Top-32
    subset, which is what guarantees that a seed has the identical colour in
    panels (b) and (c).
    """

    norm: mcolors.Normalize
    cmap: mcolors.Colormap
    values: np.ndarray          # (N,) transformed scores (log10 for 'log*')
    raw: np.ndarray             # (N,) raw scores, for the metadata only
    label: str = 'entropy-gradient score'

    def colours(self, scores: Optional[np.ndarray] = None) -> np.ndarray:
        """``(M, 3)`` float RGB in ``[0, 1]`` for the given (raw) scores.

        ``None`` returns the colours of the full seed set, in seed order.
        """
        vals = self.values if scores is None else self._transform(scores)
        return np.asarray(self.cmap(self.norm(vals)))[..., :3]

    def _transform(self, scores) -> np.ndarray:
        s = np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self.label.startswith('log10'):
            floor = max(float(s.max()) * 1e-12, 1e-30)
            s = np.log10(np.maximum(s, floor))
        return s

    @property
    def mappable(self) -> cm.ScalarMappable:
        sm = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array(np.asarray(self.values))
        return sm

from matplotlib.colors import LinearSegmentedColormap

colors = [
    '#2166AC',  # 深蓝
    '#4393C3',  # 蓝
    '#92C5DE',  # 浅蓝
    '#FEE090',  # 浅黄
    '#FDAE61',  # 橙
    '#F46D43',  # 橘红
    '#D73027',  # 红
]

custom_cmap = LinearSegmentedColormap.from_list(
    'blue_yellow_orange_red',
    colors,
    N=256
)

def build_score_colorizer(scores: Sequence[float], cmap: str = 'coolwarm',
                          mode: str = 'robust',
                          q_low: float = 1.0, q_high: float = 99.0) -> ScoreColorizer:
    """Build the score -> colour mapping of one sample.

    Args:
        scores: ``(N,)`` raw entropy-gradient magnitudes.
        cmap: matplotlib colormap name ('coolwarm': low = blue, high = red).
        mode: ``'minmax'`` (plain min/max) or ``'robust'`` (percentile clipped)
            or ``'log'`` / ``'log_robust'`` (``log10`` of the score).  The
            transform is monotone, so it never changes the ordering / Top-32.
        q_low, q_high: percentiles for the ``*robust`` modes.

    The colour scale is defined **once per sample** and then reused everywhere.
    Degenerate inputs (all scores equal, non finite values) do not divide by
    zero: every seed then simply gets the low end of the colour map.
    """
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if mode not in ('minmax', 'robust', 'log', 'log_robust'):
        raise ValueError(f'unknown score normalisation mode {mode!r}')

    values = s
    label = 'entropy-gradient score'
    if mode in ('log', 'log_robust'):
        floor = max(float(s.max()) * 1e-12, 1e-30)
        values = np.log10(np.maximum(s, floor))
        label = 'log10(entropy-gradient score)'

    if mode in ('robust', 'log_robust') and values.size:
        vmin, vmax = np.percentile(values, q_low), np.percentile(values, q_high)
    else:
        vmin, vmax = float(values.min()), float(values.max())

    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        # constant score: avoid a zero-width range (all colours at the low end)
        vmin, vmax = float(values.min()), float(values.min()) + 1.0
        if not np.isfinite(vmin):
            vmin, vmax = 0.0, 1.0

    return ScoreColorizer(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
        cmap=custom_cmap,
        values=values,
        raw=s,
        label=label,
    )


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """A single camera shared by the four panels of one sample.

    ``zoom`` is applied by both backends (``>1`` moves the eye closer), which
    lets :func:`autofit_camera` compensate for the exact field-of-view
    convention of the renderer without touching the viewing direction.
    """

    center: np.ndarray
    eye: np.ndarray
    up: np.ndarray
    fov: float = 60.0
    radius: float = 1.0
    zoom: float = 1.0


_TAN_V_CACHE = {}


def calibrate_tan_v(width: int, height: int, fov: float) -> float:
    """Measure ``tan(fov / 2)`` as the renderer actually applies it.

    Open3D does not use the requested ``fov`` verbatim (the perspective matrix
    is built from the viewport), so the mapping is measured once with two probe
    points at a known depth; the resulting value is reused for every sample.
    Falls back to the textbook value when offscreen rendering is unavailable.
    """
    key = (int(width), int(height), round(float(fov), 3))
    if key in _TAN_V_CACHE:
        return _TAN_V_CACHE[key]

    depth, offset = 10.0, 3.0
    probe = Panel()
    probe.add_points(np.array([[0.0, -offset, 0.0], [0.0, offset, 0.0]]),
                     np.zeros((2, 3)), 3.0)
    cam = Camera(center=np.zeros(3), eye=np.array([0.0, 0.0, depth]),
                 up=np.array([0.0, 1.0, 0.0]), fov=float(fov))
    try:
        image = render_open3d(probe, cam, width, height)
        box = content_bbox(image)
        half_pixels = 0.5 * (box[3] - box[2]) if box is not None else 0.0
        if half_pixels < 1.0:
            raise RuntimeError('calibration probe not visible')
        tan_eff = (offset / depth) * (height / 2.0) / half_pixels
    except Exception as exc:                      # pragma: no cover
        print(f'[viz] fov calibration failed ({exc}); using the nominal value')
        tan_eff = math.tan(math.radians(float(fov)) * 0.5)

    print(f'[viz] renderer fov: requested {fov:.1f} deg -> '
          f'effective 2*atan = {2.0 * math.degrees(math.atan(tan_eff)):.1f} deg')
    _TAN_V_CACHE[key] = tan_eff
    return tan_eff


def camera_from_points(points_xyz: Sequence[float],
                       direction: Sequence[float] = (0.8, -1.0, 0.62),
                       up: Sequence[float] = (0.0, 0.0, 1.0),
                       fov: float = 60.0, margin: float = 1.06,
                       aspect: float = 1.0,
                       tan_v: Optional[float] = None) -> Camera:
    """Fit a camera that frames ``points_xyz`` tightly from a fixed direction.

    The distance is computed by projecting every scene point into the camera
    frame and asking for the smallest distance at which all of them fall inside
    both the vertical and the horizontal field of view.  This keeps the point
    cloud large in the picture (a bounding-sphere fit wastes most of the panel
    for the flat ScanNet scenes).

    Args:
        points_xyz: ``(N, 3)`` scene points used for the fit.
        direction: viewing direction (from the scene towards the eye).
        up: up vector.
        fov: vertical field of view in degrees.
        margin: >1 leaves a small border.
        aspect: panel width / height.
        tan_v: measured ``tan(fov / 2)`` of the renderer (see
            :func:`calibrate_tan_v`); ``None`` uses the nominal value.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) * 0.5

    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    d = d / (n if n > 1e-9 else 1.0)
    up_v = np.asarray(up, dtype=np.float64).reshape(3)
    if abs(float(np.dot(d, up_v))) > 0.999 * np.linalg.norm(up_v):
        up_v = np.array([1.0, 0.0, 0.0])          # avoid a degenerate basis

    forward = -d                                  # eye -> scene
    right = np.cross(forward, up_v)
    right /= max(np.linalg.norm(right), 1e-9)
    up_c = np.cross(right, forward)

    local = pts - center[None, :]
    x = local @ right
    y = local @ up_c
    z = local @ forward
    tan_v_eff = float(tan_v) if tan_v is not None \
        else math.tan(math.radians(float(fov)) * 0.5)
    tan_h_eff = tan_v_eff * max(float(aspect), 1e-3)
    dist = float(np.max(np.maximum(np.abs(x) / max(tan_h_eff, 1e-6) - z,
                                   np.abs(y) / max(tan_v_eff, 1e-6) - z)))
    radius = float(max(np.linalg.norm(local, axis=1).max(), 1e-3))
    dist = max(dist * float(margin), radius * 0.5, 1e-3)
    return Camera(
        center=center,
        eye=center + d * dist,
        up=up_v,
        fov=float(fov),
        radius=radius,
    )


def content_bbox(image: np.ndarray, background: Sequence[float] = (255, 255, 255),
                 tol: int = 10):
    """``(x0, x1, y0, y1)`` bounding box of the non-background pixels."""
    diff = np.abs(image.astype(np.int16)
                  - np.asarray(background, dtype=np.int16)[None, None, :]).max(-1)
    ys, xs = np.nonzero(diff > tol)
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def autofit_camera(camera: Camera, points_xyz: Sequence[float],
                   width: int, height: int, target: float = 0.88,
                   iters: int = 4, tol: float = 0.04,
                   tan_v: Optional[float] = None) -> Camera:
    """Zoom ``camera`` until ``points_xyz`` fills ~``target`` of the panel.

    This is only a *verification* of :func:`camera_from_points` (which already
    uses the measured field of view): a probe cloud is rendered, the pixel
    bounding box of the content is compared to the panel size, and the zoom is
    corrected if needed.  Content that touches the image border is never
    trusted (it is clipped), the zoom is then reduced instead.  The viewing
    direction is never changed, so all four panels keep the same camera.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return camera
    probe = Panel()
    probe.add_points(pts, np.full((pts.shape[0], 3), 0.25), 2.0)

    tan_v_eff = float(tan_v) if tan_v is not None \
        else math.tan(math.radians(float(camera.fov)) * 0.5)
    aspect = float(width) / float(height)

    for _ in range(max(int(iters), 1)):
        image, _backend = render_panel(probe, camera, width, height)
        box = content_bbox(image, background=(255, 255, 255))
        if box is None:
            break

        # --- re-centre: perspective makes the bbox centre drift ----------
        delta = np.asarray(camera.eye, dtype=np.float64) - camera.center
        dist = max(float(np.linalg.norm(delta)) / max(camera.zoom, 1e-3), 1e-6)
        forward = -delta / max(np.linalg.norm(delta), 1e-9)
        right = np.cross(forward, camera.up)
        right /= max(np.linalg.norm(right), 1e-9)
        up_c = np.cross(right, forward)
        dx = (0.5 * (box[0] + box[1]) - width * 0.5) / (width * 0.5)
        dy = (0.5 * (box[2] + box[3]) - height * 0.5) / (height * 0.5)
        shift = right * (dx * tan_v_eff * aspect * dist) \
            - up_c * (dy * tan_v_eff * dist)
        if np.linalg.norm(shift) > 1e-4:
            camera.center = camera.center + shift
            camera.eye = camera.eye + shift

        clipped = (box[0] <= 1 or box[1] >= width - 2
                   or box[2] <= 1 or box[3] >= height - 2)
        frac = max((box[1] - box[0]) / float(width),
                   (box[3] - box[2]) / float(height))
        if clipped:
            camera.zoom = float(np.clip(camera.zoom * 0.85, 0.2, 20.0))
            continue
        if frac < 1e-3:
            break
        scale = float(target) / frac
        if abs(scale - 1.0) < tol and max(abs(dx), abs(dy)) < 0.03:
            break
        camera.zoom = float(np.clip(camera.zoom * scale, 0.2, 20.0))
    return camera


# --------------------------------------------------------------------------- #
# panel description (backend independent)
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Everything that has to be drawn in one sub-figure.

    ``points`` holds ``(xyz (N,3), rgb (N,3) in [0, 1], point size px)`` tuples,
    ``boxes`` holds ``(center (3,), size (3,), heading, rgb, line width)``
    tuples.  No sampling happens in here: the arrays are used verbatim.
    """

    title: str = ''
    points: List[Tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    boxes: List[Tuple[np.ndarray, np.ndarray, float, RGB, float]] = field(default_factory=list)
    background: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def add_points(self, xyz, rgb, size: float) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        rgb = np.broadcast_to(np.asarray(rgb, dtype=np.float64), xyz.shape).copy()
        self.points.append((xyz, np.clip(rgb, 0.0, 1.0), float(size)))

    def add_box(self, center, size, heading: float, rgb: RGB,
                line_width: float = 3.0) -> None:
        self.boxes.append((
            np.asarray(center, dtype=np.float64).reshape(3),
            np.asarray(size, dtype=np.float64).reshape(3),
            float(heading),
            tuple(float(c) for c in rgb),
            float(line_width),
        ))


# --------------------------------------------------------------------------- #
# Open3D offscreen rendering
# --------------------------------------------------------------------------- #
_RENDERERS = {}
_OPEN3D_FAILED = False


def _get_renderer(width: int, height: int):
    """Cached ``OffscreenRenderer`` for a given panel size."""
    global _OPEN3D_FAILED
    if o3d is None or _OPEN3D_FAILED:
        return None
    key = (int(width), int(height))
    if key not in _RENDERERS:
        try:                                        # pragma: no cover - needs GL
            renderer = o3d.visualization.rendering.OffscreenRenderer(*key)
            renderer.scene.view.set_post_processing(False)
            _RENDERERS[key] = renderer
        except Exception as exc:
            print(f'[viz] Open3D OffscreenRenderer unavailable ({exc}); '
                  f'falling back to matplotlib')
            _OPEN3D_FAILED = True
            return None
    return _RENDERERS[key]


def render_open3d(panel: Panel, camera: Camera, width: int, height: int) -> np.ndarray:
    """Render one panel with Open3D; raises if the renderer is unavailable."""
    renderer = _get_renderer(width, height)
    if renderer is None:
        raise RuntimeError('Open3D offscreen rendering not available')

    renderer.scene.clear_geometry()
    renderer.scene.set_background(list(panel.background))

    for i, (xyz, rgb, size) in enumerate(panel.points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz))
        cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(rgb))
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = float(size)
        renderer.scene.add_geometry(f'points_{i}', cloud, mat)

    for j, (center, size, heading, rgb, lw) in enumerate(panel.boxes):
        corners = box_corners(center, size, heading)
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(corners)
        lineset.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(rgb, dtype=np.float64), (len(_BOX_EDGES), 1)))
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'unlitLine'
        mat.line_width = float(lw)
        renderer.scene.add_geometry(f'box_{j}', lineset, mat)

    eye = camera.center + (np.asarray(camera.eye) - camera.center) \
        / max(float(camera.zoom), 1e-3)
    renderer.setup_camera(float(camera.fov), camera.center, eye, camera.up)
    image = renderer.render_to_image()
    return np.asarray(image)[..., :3]


# --------------------------------------------------------------------------- #
# matplotlib fallback (same panel description, 3D projection)
# --------------------------------------------------------------------------- #
def render_mpl(panel: Panel, camera: Camera, width: int, height: int,
               dpi: int = 100) -> np.ndarray:
    """Matplotlib 3D fallback used when Open3D cannot open a GL context."""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(panel.background[:3])
    ax.set_facecolor(panel.background[:3])
    ax.set_axis_off()

    eye = np.asarray(camera.eye, dtype=np.float64)
    center = np.asarray(camera.center, dtype=np.float64)
    d = center - eye
    elev = math.degrees(math.asin(np.clip(d[2] / (np.linalg.norm(d) + 1e-12), -1, 1)))
    azim = math.degrees(math.atan2(d[1], d[0]))
    ax.view_init(elev=elev, azim=azim)

    for xyz, rgb, size in panel.points:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[tuple(c) for c in rgb],
                   s=float(size) * 1.2, marker='o', linewidths=0, depthshade=False)

    for center_b, size_b, heading, rgb, lw in panel.boxes:
        seg = box_outline(center_b, size_b, heading)
        for p, q in seg:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                    color=tuple(rgb), linewidth=float(lw))

    r = max(float(camera.radius), 1e-3) / max(float(camera.zoom), 1e-3)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_box_aspect((1, 1, 1))

    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(buffer[..., :3])


def render_panel(panel: Panel, camera: Camera, width: int, height: int) -> Tuple[np.ndarray, str]:
    """Render one panel, preferring Open3D and falling back to matplotlib.

    Returns:
        ``(image (H, W, 3) uint8, backend name)``
    """
    try:
        return render_open3d(panel, camera, width, height), 'open3d'
    except Exception as exc:
        print(f'[viz] Open3D render failed ({exc}); using matplotlib for this panel')
        return render_mpl(panel, camera, width, height), 'matplotlib'
