#!/usr/bin/env python3
"""Leaf image transformation tool (Leaffliction - Part 3).

Single image  -> displays the transformations on screen.
Directory     -> saves the transformations to the destination directory.
"""
import argparse
import os
import sys

import cv2  # openCV (lecture/écriture d'images, dessin)
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
# librairie de traitement d'image dédiée aux plantes

pcv.params.debug = None
# désactive le mode debug de plantcv
# (qui sinon sauvegarde des images à chaque étape sur disque)

TRANSFORMATIONS = ["blur", "mask", "roi", "analyze", "landmarks", "histogram"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_image(path):
    # (img, path, img_name) -> recup que l'image :
    bgr_img, _, _ = pcv.readimage(filename=path)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # convertit BGR (blue green red) de opencv en RGB


def save_image(path, rgb_img):
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr_img)  # attend du BGR


def compute_mask(rgb_img):
    # canal "a" (LAB) : feuille (vert) valeur basse, fond valeur neutre
    gray = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="a")
    # seuil auto pour séparer les pixels sombres (feuille) du reste
    binary = pcv.threshold.otsu(gray_img=gray, object_type="dark")
    # supprime les petits résidus de bruit du seuillage
    filled = pcv.fill(bin_img=binary, size=50)
    # numérote le(s) objet(s) détecté(s) dans le masque (la feuille)
    labeled_mask, n_obj = pcv.create_labels(mask=filled)
    return filled, labeled_mask, n_obj
    # n_obj = nombre d'objets trouvés


def make_blur(mask):
    blurred = pcv.gaussian_blur(img=mask, ksize=(5, 5))
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)


def make_mask(rgb_img, mask):
    return pcv.apply_mask(img=rgb_img, mask=mask, mask_color="white")
    # pixels où mask == 0 (le fond) -> blancs
    # pixels où mask != 0 (la feuille) -> couleur d'origine


def make_roi(rgb_img, mask):
    out = rgb_img.copy()
    # pour tous pixels où masque non-nul = force
    # couleur à vert pur (0,255,0) en RGB :
    out[mask > 0] = (0, 255, 0)
    # sur tuple (hauteur, largeur, canaux) prends les 2 premiers
    h, w = mask.shape[:2]
    # cadre bleu :
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness=3)
    return out


def make_analyze(rgb_img, labeled_mask, n_obj):
    return pcv.analyze.size(img=rgb_img,
                            labeled_mask=labeled_mask, n_labels=n_obj)


def make_landmarks(rgb_img, mask):
    out = rgb_img.copy()
    # découpe contour de feuille en segments
    # renvoie 3 groupes de points : top, bottom, center_v
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=rgb_img,
                                                                mask=mask)
    for points, color in ((top, (255, 0, 0)),
                          (bottom, (255, 0, 255)), (center_v, (255, 140, 0))):
        for point in points:
            x, y = point[0]
            # point[0] : chaque point encapsulé dans tab [[x, y]]
            cv2.circle(out, (int(x), int(y)), 3, color, -1)  # -1 = rempli
    return out


CHANNELS = [
    ("blue", "blue_frequencies", "blue"),
    ("blue-yellow", "blue-yellow_frequencies", "yellow"),
    ("green", "green_frequencies", "green"),
    ("green-magenta", "green-magenta_frequencies", "magenta"),
    ("hue", "hue_frequencies", "purple"),
    ("lightness", "lightness_frequencies", "gray"),
    ("red", "red_frequencies", "red"),
    ("saturation", "saturation_frequencies", "cyan"),
    ("value", "value_frequencies", "orange"),
]
# nom affiché dans légende → clé dans résultats plantcv
# → couleur courbe sur graphique


def make_histogram_figure(rgb_img, labeled_mask, n_obj):
    pcv.outputs.clear()  # pas accumuler results img précédente
    pcv.analyze.color(rgb_img=rgb_img, labeled_mask=labeled_mask,
                      n_labels=n_obj, colorspaces="all")
    observations = next(iter(pcv.outputs.observations.values()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, key, color in CHANNELS:
        values = observations[key]["value"]
        x = np.linspace(0, 255, num=len(values))
        ax.plot(x, values, label=name, color=color, linewidth=1)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend(title="color Channel", loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def build_transformations(rgb_img, selected):
    mask, labeled_mask, n_obj = compute_mask(rgb_img)
    results = {}
    if "blur" in selected:
        results["Blur"] = make_blur(mask)
    if "mask" in selected:
        results["Mask"] = make_mask(rgb_img, mask)
    if "roi" in selected:
        results["Roi"] = make_roi(rgb_img, mask)
    if "analyze" in selected:
        results["Analyze"] = make_analyze(rgb_img, labeled_mask, n_obj)
    if "landmarks" in selected:
        results["Landmarks"] = make_landmarks(rgb_img, mask)
    if "histogram" in selected:
        results["Histogram"] = make_histogram_figure(rgb_img,
                                                     labeled_mask, n_obj)
    return results


def display_single_image(path, selected):
    rgb_img = read_image(path)
    results = build_transformations(rgb_img, selected)

    # sépare histogramme qui n'est pas img
    image_results = {k: v for k, v in results.items() if k != "Histogram"}
    n = len(image_results) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, (name, img) in zip(axes[1:], image_results.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# parcourt récursivement src_dir (et sous-dossiers)
# renvoie un par un les chemins de fichiers image, triés par nom
def find_images(src_dir):
    for root, _, files in os.walk(src_dir):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, name)


# crée dossier dest s'il existe pas
# puis pour chaque img calcul transfos et les sauvegarde
def process_directory(src_dir, dst_dir, selected):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for path in find_images(src_dir):
        rgb_img = read_image(path)
        results = build_transformations(rgb_img, selected)
        base, ext = os.path.splitext(os.path.basename(path))
        for name, img in results.items():
            out_path = os.path.join(dst_dir, f"{base}_{name}{ext}")
            if name == "Histogram":
                img.savefig(out_path)
                plt.close(img)
            else:
                save_image(out_path, img)
        count += 1
    print(f"Transformed {count} image(s) into '{dst_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply leaf image transformations (Leaffliction Part 3).",
    )
    parser.add_argument("image", nargs="?",
                        help="Path to a single image to display.")
    parser.add_argument("-src", dest="src", help="Source directory of images.")
    parser.add_argument("-dst", dest="dst",
                        help="Destination directory for transformed images.")
    for name in TRANSFORMATIONS:
        parser.add_argument(f"-{name}", action="store_true",
                            help=f"Only generate the {name} transformation.")
    return parser.parse_args()


def main():
    args = parse_args()
    active = [name for name in TRANSFORMATIONS if getattr(args, name)]
    selected = active or TRANSFORMATIONS

    if args.image:
        display_single_image(args.image, selected)
        return

    if args.src and args.dst:
        process_directory(args.src, args.dst, selected)
        return

    print(
        "Usage: ./Transformation.py <image> | "
        "-src <dir> -dst <dir> "
        "[-blur|-mask|-roi|-analyze|-landmarks|-histogram]",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
