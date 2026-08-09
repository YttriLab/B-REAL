"""Conversions between numpy RGB frames (from VideoReader) and QImage.

The decoder yields (H, W, 3) uint8 RGB arrays; Qt paints QImage. These helpers
are the seam between the two, and are unit-tested for exactness (a frame that
round-trips array -> QImage -> array must be pixel-identical).
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtGui


def ndarray_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """(H, W, 3) uint8 RGB -> QImage (Format_RGB888).

    Returns a QImage that owns its pixels (``.copy()``), so the source numpy
    buffer can be freed/reused without dangling the image.
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return img.copy()


def qimage_to_ndarray(img: QtGui.QImage) -> np.ndarray:
    """QImage -> (H, W, 3) uint8 RGB array (handles row padding)."""
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(memoryview(img.constBits()), dtype=np.uint8)
    buf = buf.reshape((h, bpl))[:, : w * 3]  # drop any per-row padding
    return buf.reshape((h, w, 3)).copy()
