"""Leaf / background segmentation shared by Transformation, train, predict.

The background is modelled instead of the leaf: segmenting the green of
the leaf drops the brown and black disease spots, which are exactly
what the model must learn. The background of the data set is a fairly
uniform grey / purple surface, so every pixel far enough from the
colors seen on the image borders is considered to belong to the leaf.
"""
import cv2
import numpy as np

BORDER_WIDTH = 8  # épaisseur (px) de la bande de bord = échantillon du fond
BACKGROUND_CLUSTERS = 4  # plusieurs couleurs de fond (éclairage non uniforme)
# poids de la luminosité (L) dans la distance LAB : faible pour que les
# ombres, qui changent surtout L, restent proches du fond
LIGHTNESS_WEIGHT = 0.3
OTSU_FACTOR = 0.6  # seuil plus bas qu'Otsu pour ne pas trouer la feuille
WHITE_LEVEL = 250  # remplissage blanc des augmentations = fond
GREEN_A_LIMIT = 124  # canal a (LAB) sous 128 = vert ; marge pour le gris
MIN_COMPONENT_RATIO = 0.02  # morceaux < 2% de l'image = bruit
MIN_LEAF_RATIO = 0.05  # masque hors [5%, 95%] = segmentation ratée
MAX_LEAF_RATIO = 0.95


def _border_pixels(img):
    b = BORDER_WIDTH
    return np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3),
                           img[:, -b:].reshape(-1, 3)])


def _is_white(rgb_img):
    return rgb_img.min(axis=-1) > WHITE_LEVEL


def _background_colors(rgb_img, lab_img):
    border = _border_pixels(lab_img)
    # ignore le blanc laissé par Rotate/Shear/... s'il reste assez de fond
    not_white = ~_is_white(_border_pixels(rgb_img))
    if not_white.sum() >= 10 * BACKGROUND_CLUSTERS:
        border = border[not_white]
    k = min(BACKGROUND_CLUSTERS, len(border))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(border, k, None, criteria, 2,
                               cv2.KMEANS_PP_CENTERS)
    # le fond n'est jamais vert : si la feuille touche les bords, ses
    # couleurs (a < 128 en LAB = vert) ne doivent pas passer pour du fond
    not_green = centers[centers[:, 1] >= GREEN_A_LIMIT]
    return not_green if len(not_green) else centers


def _background_distance(rgb_img):
    # distance LAB de chaque pixel à la couleur de fond la plus proche,
    # ramenée sur 0-255 pour pouvoir seuiller avec Otsu
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    weights = np.array([LIGHTNESS_WEIGHT, 1, 1], np.float32)
    dist = np.min([np.sqrt((((lab - c) * weights) ** 2).sum(axis=-1))
                   for c in _background_colors(rgb_img, lab)], axis=0)
    dist = dist * 255 / max(float(dist.max()), 1.0)
    return dist.astype(np.uint8)


def _keep_large_components(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas)
            if a >= MIN_COMPONENT_RATIO * mask.size]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def _fill_holes(mask):
    # remplit l'intérieur des contours : les taches de maladie au milieu
    # de la feuille font partie de la feuille
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment_leaf(rgb_img):
    """Return the leaf mask (uint8, 0 = background, 255 = leaf).

    Raises ValueError when the result does not look like a leaf.
    """
    dist = _background_distance(rgb_img)
    otsu, _ = cv2.threshold(dist, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.where(dist > otsu * OTSU_FACTOR, 255, 0).astype(np.uint8)
    mask[_is_white(rgb_img)] = 0
    # ouverture : retire le bruit ; fermeture : recolle nervures / reflets
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = _fill_holes(_keep_large_components(mask))
    ratio = np.count_nonzero(mask) / mask.size
    if not MIN_LEAF_RATIO <= ratio <= MAX_LEAF_RATIO:
        raise ValueError(f"no leaf detected in image "
                         f"(mask covers {ratio:.0%} of it)")
    return mask


def apply_leaf_mask(rgb_img, mask):
    """Return the image with the background painted white."""
    out = rgb_img.copy()
    out[mask == 0] = 255
    return out


def prepare_for_model(rgb_img):
    """Return (image fed to the model, True if the leaf was segmented).

    Falls back on the raw image when the segmentation fails, so that
    training and prediction always get an image to work on.
    """
    try:
        return apply_leaf_mask(rgb_img, segment_leaf(rgb_img)), True
    except ValueError:
        return rgb_img, False
