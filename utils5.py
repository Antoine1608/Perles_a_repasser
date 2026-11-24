"""utils5 — fonctions de traitement d'image pour l'appli « perles à repasser ».

Pipeline (inspiré de notebooks/main_v0.ipynb) :
    photo -> rotation 90° x k -> recadrage carré -> max-pooling 2x2 répété
          -> quantification des couleurs (K-Means, fallback PIL)
          -> rendu PIL : chaque pixel devient une perle ronde (trou gris au centre)

Dépendances : numpy + Pillow (obligatoires), scikit-learn (optionnel).
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("perles")

# Extensions reconnues pour la galerie
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")


# --------------------------------------------------------------------------- #
#  Chargement / rotation                                                        #
# --------------------------------------------------------------------------- #
def load_image(source) -> np.ndarray:
    """Charge une image (chemin, fichier ouvert ou bytes) en tableau RGB uint8 (H, W, 3)."""
    img = Image.open(source).convert("RGB")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"format d'image incorrect : shape={arr.shape}")
    logger.debug("load_image -> shape=%s dtype=%s", arr.shape, arr.dtype)
    return arr


def rotate_image(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotation de 90° × k dans le sens anti-horaire (k = 0, 1, 2, 3)."""
    return np.rot90(img, k=k % 4)


# --------------------------------------------------------------------------- #
#  Réduction au canvas carré (max-pooling 2x2, comme MaxPool2D du notebook)     #
# --------------------------------------------------------------------------- #
def _maxpool2(img: np.ndarray) -> np.ndarray:
    """Max-pooling 2x2 (les bords impairs sont rognés)."""
    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    img = img[:h2, :w2]
    return img.reshape(h2 // 2, 2, w2 // 2, 2, *img.shape[2:]).max(axis=(1, 3))


def resize_to_canvas(img: np.ndarray, size: int) -> np.ndarray:
    """Recadre l'image au carré (centre) puis la réduit par max-pooling 2x2
    répété jusqu'à approcher `size`, et ajuste exactement à size x size."""
    if size < 1:
        raise ValueError("size doit être >= 1")
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    img = img[top:top + s, left:left + s]

    # nombre de pools 2x2 pour passer sous `size` (comme ceil(log2(...)) du notebook,
    # mais en restant déterministe et sans dépasser)
    k = int(np.floor(np.log2(s / size))) if s > size else 0
    k = max(0, k)
    for i in range(k):
        img = _maxpool2(img)
        logger.debug("maxpool #%d -> %s", i + 1, img.shape)

    if img.shape[0] != size or img.shape[1] != size:
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.resize((size, size), Image.NEAREST)
        img = np.asarray(pil)
    return img


# --------------------------------------------------------------------------- #
#  Quantification des couleurs                                                 #
# --------------------------------------------------------------------------- #
def reduce_colors(img: np.ndarray, n_colors: int) -> np.ndarray:
    """Réduit l'image à `n_colors` couleurs.

    K-Means (scikit-learn) si disponible, sinon quantification médiane de PIL.
    Retourne un tableau uint8 de même shape que `img`.
    """
    pixels = img.reshape(-1, 3)
    n = int(min(max(1, n_colors), len(pixels)))
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n, n_init=10, random_state=42)
        labels = km.fit_predict(pixels.astype(np.float64))
        centers = km.cluster_centers_
        logger.debug("KMeans: %d clusters sur %d pixels", n, len(pixels))
    except ImportError:
        logger.debug("scikit-learn indisponible -> fallback PIL quantize(%d)", n)
        q = Image.fromarray(img.astype(np.uint8)).quantize(colors=n)
        palette = np.asarray(q.getpalette()[: n * 3], dtype=np.uint8).reshape(-1, 3)
        labels = np.asarray(q).reshape(-1)
        centers = palette.astype(np.float64)
    out = centers[labels].reshape(img.shape).astype(np.uint8)
    return out


def build_quantized(img: np.ndarray, size: int, n_colors: int):
    """Pipeline complet : resize canvas + quantification.
    Retourne (small, quant) : image réduite et image quantifiée (uint8, size x size x 3)."""
    small = resize_to_canvas(img, size)
    quant = reduce_colors(small, n_colors)
    return small, quant


def palette(quant: np.ndarray) -> np.ndarray:
    """Couleurs réellement utilisées, triées (R, G, B) — pour la légende."""
    return np.unique(quant.reshape(-1, 3), axis=0)


def to_hex(rgb) -> str:
    """(r, g, b) -> '#RRGGBB'."""
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


# --------------------------------------------------------------------------- #
#  Rendu « perles à repasser »                                                 #
# --------------------------------------------------------------------------- #
def _bead_tile(color: tuple, scale: int, hole_ratio: float = 0.18) -> Image.Image:
    """Une perle = disque de couleur inscrit dans un carré `scale` px,
    avec un petit trou gris au centre (comme le cercle gris du notebook)."""
    tile = Image.new("RGB", (scale, scale), "white")
    d = ImageDraw.Draw(tile)
    d.ellipse([0, 0, scale - 1, scale - 1], fill=color)
    hr = max(1.0, scale * hole_ratio)
    c = (scale - 1) / 2.0
    d.ellipse([c - hr, c - hr, c + hr, c + hr], fill=(150, 150, 150))
    return tile


def render_beads(quant: np.ndarray, scale: int = 10) -> Image.Image:
    """Dessine la grille de perles : chaque pixel de `quant` devient une perle.

    Optimisation : une tuile est dessinée une seule fois par couleur puis collée.
    """
    h, w = quant.shape[:2]
    out = Image.new("RGB", (w * scale, h * scale), "white")
    tiles: dict[tuple, Image.Image] = {}
    for i in range(h):
        row = quant[i]
        y = i * scale
        for j in range(w):
            c = (int(row[j][0]), int(row[j][1]), int(row[j][2]))
            tile = tiles.get(c)
            if tile is None:
                tile = tiles[c] = _bead_tile(c, scale)
            out.paste(tile, (j * scale, y))
    logger.debug("render_beads: %dx%d perles, échelle %d -> %s", h, w, scale, out.size)
    return out


# --------------------------------------------------------------------------- #
#  Export                                                                       #
# --------------------------------------------------------------------------- #
def grid_to_csv(quant: np.ndarray) -> str:
    """CSV 'ligne,colonne,hex' (index 1-based) de toute la grille de perles."""
    buf = io.StringIO()
    buf.write("ligne,colonne,hex,r,g,b\n")
    h, w = quant.shape[:2]
    for i in range(h):
        for j in range(w):
            r, g, b = (int(v) for v in quant[i, j])
            buf.write(f"{i + 1},{j + 1},#{r:02X}{g:02X}{b:02X},{r},{g},{b}\n")
    return buf.getvalue()
