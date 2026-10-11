"""Multi-view consistent licence-plate replacement.

Per-frame blurring flickers, misses plates at grazing angles, and leaves a smeared plate baked into the 3D
model. Instead, plates are located in 3D once and re-projected into every view:

  1. detect plates per frame (any detector; default: a YOLOv8 plate model via ultralytics, optional);
  2. refine each box to the plate's 4 corners (largest convex quadrilateral inside the box);
  3. group detections by physical plate, using the SfM points that fall inside each box;
  4. triangulate each corner over all detections (DLT, outlier views dropped) and square it up into a
     planar rectangle;
  5. project the rectangle into every frame that faces it, including frames where detection failed;
  6. warp a replacement plate in with a homography, matched to the local brightness and edge-feathered.

The edited images replace the dataset images, so the trained splat carries the new plate from every angle.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from v2s import colmap_io as cio

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "plate-yolov8.pt"


@dataclass
class Detection:
    image: str
    box: np.ndarray          # x1, y1, x2, y2
    corners: np.ndarray      # (4, 2) TL, TR, BR, BL
    score: float
    center3d: np.ndarray | None = None


@dataclass
class Plate:
    corners3d: np.ndarray    # (4, 3) TL, TR, BR, BL
    normal: np.ndarray       # faces the cameras that saw it
    detections: list = field(default_factory=list)
    reproj_px: float = 0.0


# ---------------------------------------------------------------------------------------------- detection
def detect(images: list[Path], weights: Path = DEFAULT_WEIGHTS, conf: float = 0.3) -> list[Detection]:
    from ultralytics import YOLO   # optional dependency (AGPL-3.0); swap in any detector returning boxes

    model = YOLO(str(weights))
    dets = []
    for p in images:
        img = cv2.imread(str(p))
        for b in model.predict(img, conf=conf, verbose=False)[0].boxes:
            box = b.xyxy[0].cpu().numpy()
            dets.append(Detection(p.name, box, refine_corners(img, box), float(b.conf[0])))
    return dets


def order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=float)


def refine_corners(img: np.ndarray, box: np.ndarray, pad: float = 0.15) -> np.ndarray:
    """Plate corners inside a detection box; falls back to the box itself."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    X2, Y2 = int(min(img.shape[1], x2 + pad * w)), int(min(img.shape[0], y2 + pad * h))
    crop = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)
    fallback = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
    if crop.size == 0:
        return fallback
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 40, 120), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.35 * w * h        # must cover a good part of the box
    for c in contours:
        quad = cv2.approxPolyDP(cv2.convexHull(c), 0.04 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(quad)
        if len(quad) == 4 and best_area < area < 1.6 * w * h:
            best, best_area = quad[:, 0, :].astype(float) + [X1, Y1], area
    return order_corners(best) if best is not None else fallback


# ---------------------------------------------------------------------------------------------- geometry
def _P(cam: cio.Camera, im: cio.Image) -> np.ndarray:
    return cam.K() @ np.c_[im.R, im.t]


def triangulate(points2d: list[np.ndarray], Ps: list[np.ndarray]) -> np.ndarray:
    A = []
    for (u, v), P in zip(points2d, Ps):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = P @ np.c_[X, np.ones(len(X))].T
    return (x[:2] / x[2]).T


def locate_plates(dets: list[Detection], cam, images: dict, xyz_by_id: dict, cluster_frac: float = 0.08,
                  min_views: int = 3) -> list[Plate]:
    by_name = {im.name: im for im in images.values()}
    centers = np.array([im.center for im in images.values()])
    scale = float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    up = -np.mean([im.R[1] for im in images.values()], axis=0)   # OpenCV y points down
    up /= np.linalg.norm(up)

    for d in dets:   # 3D centre: median of SfM points observed inside the box
        im = by_name[d.image]
        x1, y1, x2, y2 = d.box
        inside = (im.xys[:, 0] >= x1) & (im.xys[:, 0] <= x2) & (im.xys[:, 1] >= y1) & (im.xys[:, 1] <= y2)
        pts = [xyz_by_id[i] for i in im.point3D_ids[inside] if i in xyz_by_id]
        d.center3d = np.median(pts, axis=0) if len(pts) >= 3 else None

    clusters: list[list[Detection]] = []
    for d in sorted([d for d in dets if d.center3d is not None], key=lambda d: -d.score):
        for c in clusters:
            if np.linalg.norm(c[0].center3d - d.center3d) < cluster_frac * scale:
                c.append(d)
                break
        else:
            clusters.append([d])

    plates = []
    for c in clusters:
        if len(c) < min_views:
            continue
        Ps = [_P(cam, by_name[d.image]) for d in c]
        keep = list(range(len(c)))
        for _ in range(2):   # triangulate, drop views with large corner error, repeat
            X = np.array([triangulate([c[i].corners[k] for i in keep], [Ps[i] for i in keep]) for k in range(4)])
            err = np.array([np.abs(project(Ps[i], X) - c[i].corners).max() for i in range(len(c))])
            new_keep = [i for i in range(len(c)) if err[i] <= max(4.0, 2.5 * np.median(err[keep]))]
            if len(new_keep) < min_views or new_keep == keep:
                break
            keep = new_keep
        X = fit_to_boxes(rectify(X), [c[i] for i in keep], [Ps[i] for i in keep], up)
        err = np.array([np.abs(project(Ps[i], X) - c[i].corners).max() for i in keep])
        n = np.cross(X[1] - X[0], X[3] - X[0])
        n /= np.linalg.norm(n)
        view = np.mean([by_name[c[i].image].center for i in keep], axis=0) - X.mean(0)
        if n @ view < 0:
            n = -n
        plates.append(Plate(X, n, [c[i] for i in keep], float(np.median(err))))
    return plates


def _rect_from_params(p: np.ndarray) -> np.ndarray:
    """Params: centre (3), rotation vector (3), width, height -> corners TL, TR, BR, BL."""
    R = cv2.Rodrigues(p[3:6])[0]
    u, v = R[:, 0] * p[6] / 2, R[:, 1] * p[7] / 2
    c = p[:3]
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def fit_to_boxes(X0: np.ndarray, dets: list, Ps: list, up: np.ndarray | None = None) -> np.ndarray:
    """Refine the 3D rectangle so its projection matches every detection box (robust least squares).

    Detector boxes are far more stable than per-frame corner estimates; the initial rectangle only
    supplies orientation and a starting point.
    """
    from scipy.optimize import least_squares

    u = X0[1] - X0[0]
    v = X0[3] - X0[0]
    w, h = np.linalg.norm(u), np.linalg.norm(v)
    R0 = np.stack([u / w, v / h, np.cross(u / w, v / h)], axis=1)
    U, _, Vt = np.linalg.svd(R0)
    p0 = np.r_[X0.mean(0), cv2.Rodrigues(U @ Vt)[0].ravel(), w, h]

    def residuals(p):
        out = []
        for d, P in zip(dets, Ps):
            q = project(P, _rect_from_params(p))
            out += list(np.r_[q.min(0), q.max(0)] - d.box)
        return np.array(out)

    res = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
    X = _rect_from_params(res.x)
    if up is None:
        return X

    # Boxes barely constrain in-plane rotation, and plates are mounted level: make the top edge
    # horizontal (perpendicular to the scene's up axis), then refit only centre and size.
    n = np.cross(X[1] - X[0], X[3] - X[0])
    n /= np.linalg.norm(n)
    u = X[1] - X[0]
    u = u - (u @ up) * up - (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    if v @ (X[3] - X[0]) < 0:
        v = -v
    rvec = cv2.Rodrigues(np.stack([u, v, np.cross(u, v)], axis=1))[0].ravel()

    def residuals_level(q):
        return residuals(np.r_[q[:3], rvec, q[3:]])

    q = least_squares(residuals_level, np.r_[res.x[:3], res.x[6:]], loss="soft_l1", f_scale=3.0).x
    return _rect_from_params(np.r_[q[:3], rvec, q[3:]])


def rectify(X: np.ndarray, grow: float = 1.0) -> np.ndarray:
    """Closest planar rectangle to 4 triangulated corners (TL, TR, BR, BL), grown slightly to cover the frame."""
    c = X.mean(0)
    u = ((X[1] - X[0]) + (X[2] - X[3])) / 2
    v = ((X[3] - X[0]) + (X[2] - X[1])) / 2
    w, h = np.linalg.norm(u) * grow, np.linalg.norm(v) * grow
    u /= np.linalg.norm(u)
    v = v - (v @ u) * u
    v /= np.linalg.norm(v)
    return np.array([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])


# ---------------------------------------------------------------------------------------------- rendering
def plate_texture(text: str, sub: str, size=(640, 320)) -> np.ndarray:
    w, h = size
    tex = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(tex, (8, 8), (w - 9, h - 9), (40, 40, 40), 10)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.0
    while cv2.getTextSize(text, font, scale + 0.1, 8)[0][0] < w * 0.82:
        scale += 0.1
    (tw, th), _ = cv2.getTextSize(text, font, scale, 8)
    cv2.putText(tex, text, ((w - tw) // 2, (h + th) // 2 + 10), font, scale, (30, 30, 30), 8, cv2.LINE_AA)
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(tex, sub, ((w - sw) // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 60, 20), 2, cv2.LINE_AA)
    return tex


def paste(img: np.ndarray, quad: np.ndarray, tex: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    warped = cv2.warpPerspective(tex, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (img.shape[1], img.shape[0]))
    if mask.sum() == 0:
        return img
    # Match the plate's whites to the original plate's bright pixels (the dark frame/border would bias a mean).
    region = mask > 128
    bright_src = np.percentile(img[region].reshape(-1, 3).mean(1), 90)
    bright_tex = np.percentile(warped[region].reshape(-1, 3).mean(1), 90)
    warped = np.clip(warped * np.clip(bright_src / max(bright_tex, 1), 0.35, 1.15), 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255
    return (img * (1 - alpha) + warped * alpha).astype(np.uint8)


def visible(plate: Plate, im: cio.Image, max_angle_deg: float = 55) -> bool:
    # Beyond ~55 deg the bumper edge hides most of the plate and what is left is unreadable; pasting
    # there would put the new plate on top of the bumper instead.
    to_cam = im.center - plate.corners3d.mean(0)
    cosang = plate.normal @ to_cam / np.linalg.norm(to_cam)
    in_front = ((im.R @ plate.corners3d.T).T + im.t)[:, 2] > 0
    return cosang > np.cos(np.deg2rad(max_angle_deg)) and in_front.all()


def occluded(plate: Plate, im: cio.Image, quad2d: np.ndarray, xyz_by_id: dict, margin: float = 1.0,
             min_points: int = 8) -> bool:
    """True if SfM points observed in this frame sit inside the plate outline but clearly in front of it.

    margin is in plate widths: a bumper step ~10 cm in front of the plate does not count; a body panel seen
    from the side (a plate width or more in front) does.
    """
    inside = cv2.pointPolygonTest
    poly = quad2d.astype(np.float32).reshape(-1, 1, 2)
    width = np.linalg.norm(plate.corners3d[1] - plate.corners3d[0])
    plate_depth = (im.R @ plate.corners3d.mean(0) + im.t)[2]
    hits = 0
    for (x, y), pid in zip(im.xys, im.point3D_ids):
        if pid in xyz_by_id and inside(poly, (float(x), float(y)), False) >= 0:
            if (im.R @ xyz_by_id[pid] + im.t)[2] < plate_depth - margin * width:
                hits += 1
    return hits >= min_points


def grow_quad(q: np.ndarray, f: float) -> np.ndarray:
    c = q.mean(0)
    return c + (q - c) * f


# ---------------------------------------------------------------------------------------------- pipeline
def replace_plates(dataset: Path, text: str = "VIDEO2SPLAT", sub: str = "github.com/neelgajera",
                   weights: Path = DEFAULT_WEIGHTS) -> dict:
    images_dir = dataset / "images"
    backup = dataset / "images_original"
    if not backup.exists():
        shutil.copytree(images_dir, backup)
    cams, imgs, _, _ = cio.read_model(dataset / "sparse" / "0")
    cam = next(iter(cams.values()))
    xyz_by_id = read_points_by_id(dataset / "sparse" / "0")

    names = sorted(im.name for im in imgs.values())
    dets = detect([backup / n for n in names], weights)
    plates = locate_plates(dets, cam, imgs, xyz_by_id)
    tex = plate_texture(text, sub)

    edited = skipped_occluded = 0
    for im in imgs.values():
        img = cv2.imread(str(backup / im.name))
        hit = False
        for pl in plates:
            if not visible(pl, im):
                continue
            quad = project(_P(cam, im), pl.corners3d)
            if occluded(pl, im, quad, xyz_by_id):
                skipped_occluded += 1
                continue
            img = paste(img, grow_quad(quad, 1.12), tex)   # cover the original plate's frame too
            hit = True
        cv2.imwrite(str(images_dir / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        edited += hit

    detected_frames = {d.image for d in dets}
    stats = {"frames": len(names), "detections": len(dets), "frames_with_detection": len(detected_frames),
             "plates": len(plates), "frames_edited": edited, "views_skipped_occluded": skipped_occluded,
             "plate_views_used": [len(p.detections) for p in plates],
             "corner_reprojection_px": [round(p.reproj_px, 2) for p in plates]}
    (dataset / "plates.json").write_text(json.dumps(
        {**stats, "corners3d": [p.corners3d.tolist() for p in plates]}, indent=2))
    return stats


def read_points_by_id(model: Path) -> dict:
    import struct
    out = {}
    with open(model / "points3D.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            pid, x, y, z, _r, _g, _b, _e, n = struct.unpack("<QdddBBBdQ", f.read(51))
            f.seek(8 * n, 1)
            out[pid] = np.array([x, y, z])
    return out


def contact_sheet(dataset: Path, out: Path, n: int = 6, crop: int = 220) -> None:
    """Before/after crops around the replaced plate in frames that changed, for the README."""
    info = json.loads((dataset / "plates.json").read_text())
    cams, imgs, _, _ = cio.read_model(dataset / "sparse" / "0")
    cam = next(iter(cams.values()))
    tiles = []
    for X in map(np.array, info["corners3d"]):
        changed = []
        for im in sorted(imgs.values(), key=lambda i: i.name):
            a = cv2.imread(str(dataset / "images_original" / im.name))
            b = cv2.imread(str(dataset / "images" / im.name))
            if np.abs(a.astype(int) - b.astype(int)).max() > 30:
                changed.append(im)
        for im in changed[:: max(1, len(changed) // n)][:n]:
            c = project(_P(cam, im), X).mean(0).astype(int)
            pair = []
            for folder in ("images_original", "images"):
                img = cv2.copyMakeBorder(cv2.imread(str(dataset / folder / im.name)), crop, crop, crop, crop,
                                         cv2.BORDER_REFLECT)
                pair.append(img[c[1]:c[1] + 2 * crop, c[0]:c[0] + 2 * crop])
            tiles.append(np.vstack(pair))
    if tiles:
        cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
