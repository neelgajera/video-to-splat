"""Minimal COLMAP binary model reader/writer (cameras.bin, images.bin, points3D.bin).

Only what this project needs: PINHOLE / SIMPLE_PINHOLE / OPENCV cameras, poses, and colored points.
Written from the format spec in COLMAP's src/colmap/scene/reconstruction_io.cc, so no pycolmap dependency.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}
MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class Image:
    id: int
    name: str
    camera_id: int
    R: np.ndarray            # world -> camera rotation (3x3)
    t: np.ndarray            # world -> camera translation (3,)
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


def qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_qvec(R: np.ndarray) -> np.ndarray:
    # Eigenvector of the symmetric 4x4 matrix (same as COLMAP's rotmat2qvec).
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                  [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_model(path: Path) -> tuple[dict[int, Camera], dict[int, Image], np.ndarray, np.ndarray]:
    """Returns cameras, images, points xyz (N,3) and colors (N,3 uint8)."""
    path = Path(path)
    cams: dict[int, Camera] = {}
    with open(path / "cameras.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, mid, w, h = _read(f, "iiQQ")
            name, n = CAMERA_MODELS[mid]
            cams[cid] = Camera(cid, name, w, h, np.array(_read(f, "d" * n)))
    imgs: dict[int, Image] = {}
    with open(path / "images.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n = _read(f, "Q")[0]
            data = np.frombuffer(f.read(24 * n), dtype=np.dtype([("xy", "<f8", 2), ("pid", "<i8")]))
            imgs[iid] = Image(iid, name.decode(), cid, qvec_to_R(np.array([qw, qx, qy, qz])),
                              np.array([tx, ty, tz]), data["xy"].copy(), data["pid"].copy())
    xyz, rgb = [], []
    with open(path / "points3D.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            _, x, y, z, r, g, b, _err, n = _read(f, "QdddBBBdQ")
            f.seek(8 * n, 1)
            xyz.append((x, y, z)); rgb.append((r, g, b))
    return cams, imgs, np.array(xyz).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3)


def write_model(path: Path, cameras: list[Camera], images: list[Image], xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a model; points are written without tracks (enough for splatting initialisation)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for c in cameras:
            f.write(struct.pack("<iiQQ", c.id, MODEL_IDS[c.model], c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *map(float, c.params)))
    with open(path / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            q = R_to_qvec(im.R)
            f.write(struct.pack("<idddddddi", im.id, *q, *im.t, im.camera_id))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(path / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(struct.pack("<QdddBBBdQ", i, *map(float, p), *map(int, c), 0.0, 0))
