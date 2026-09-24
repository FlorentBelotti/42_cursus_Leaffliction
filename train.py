#!/usr/bin/env python3
"""Leaf disease classifier training (Leaffliction - Part 4).

Fetches images from a directory and its subdirectories (one
subdirectory per class), trains a model to recognize the diseases,
and saves the trained model together with the images used for
training into a .zip archive.
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from leaffliction.dataset_balancing.dataset_balancing_pipeline import (
    run_dataset_balancing,
)
from leaffliction.leaf_segmentation import prepare_for_model

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMG_SHAPE = (256, 256, 3)


# parcourt récursivement src_dir, renvoie (chemin, label) par image
# label = nom du sous-dossier direct parent (ex: apple_healthy)
def find_images(src_dir):
    for root, _, files in os.walk(src_dir):
        label = os.path.basename(root)
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, name), label


def load_dataset(src_dir):
    paths, labels = [], []
    for path, label in find_images(src_dir):
        paths.append(path)
        labels.append(label)
    return paths, labels


def check_dataset(labels, val_ratio):
    # train_test_split(stratify=...) plante avec un message obscur si
    # une classe a < 2 images ou si le val set est plus petit que le
    # nombre de classes : on vérifie avant avec un message clair
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(f"need at least 2 classes, found {len(classes)}")
    for name in classes:
        if labels.count(name) < 2:
            raise ValueError(f"class '{name}' needs at least 2 images "
                             "(one for training, one for validation)")
    n_val = math.ceil(len(labels) * val_ratio)
    n_train = len(labels) - n_val
    if n_val < len(classes) or n_train < len(classes):
        raise ValueError(
            f"{len(labels)} image(s) with --val-ratio {val_ratio} leaves "
            f"{n_train} for training and {n_val} for validation, "
            f"need at least {len(classes)} (one per class) in each")


def split_train_val(paths, labels, val_ratio=0.2, seed=42):
    # split stratifié : conserve la proportion de chaque classe
    # dans le train set et le val set
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_paths, train_labels, val_paths, val_labels


def copy_images(paths, labels, dst_dir):
    # recrée l'arborescence dst_dir/<label>/<image>
    for path, label in zip(paths, labels):
        class_dir = os.path.join(dst_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy2(path, class_dir)


def augment_training_set(paths, labels, dst_dir):
    # équilibrage de la Partie 2 (Florent) appliqué au seul train set :
    # aucune variante d'une image de validation ne fuit dans le train
    with tempfile.TemporaryDirectory() as staging_dir:
        copy_images(paths, labels, staging_dir)
        run_dataset_balancing(Path(staging_dir), Path(dst_dir),
                              should_overwrite_destination=True)


def segment_directory(directory):
    # remplace chaque image par sa version détourée (fond blanc), la
    # même que predict donnera au modèle ; repli sur l'image brute si
    # la feuille n'est pas trouvée
    total, fallback = 0, 0
    for path, _ in find_images(directory):
        bgr_img = cv2.imread(path)
        if bgr_img is None:
            raise ValueError(f"'{path}' is not a readable image")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        model_img, segmented = prepare_for_model(rgb_img)
        if not cv2.imwrite(path, cv2.cvtColor(model_img, cv2.COLOR_RGB2BGR)):
            raise OSError(f"cannot write '{path}'")
        total += 1
        fallback += not segmented
    print(f"Segmented {total - fallback}/{total} image(s) in '{directory}'"
          f" ({fallback} kept raw: no leaf detected)")


def load_split(directory, classes, shuffle):
    # class_names fixe l'ordre des labels (index -> nom de classe)
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=classes,
        image_size=IMG_SHAPE[:2],
        batch_size=32,
        shuffle=shuffle,
        seed=42,
    )


def build_model(num_classes, input_shape=IMG_SHAPE):
    # CNN simple : 3 blocs conv/pooling puis classifieur dense
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        # labels entiers (pas one-hot) -> sparse_categorical_crossentropy
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_data, val_data, epochs=10):
    return model.fit(train_data, validation_data=val_data, epochs=epochs)


def save_bundle(model, classes, images_dir, out_zip):
    # regroupe modèle entraîné + noms des classes + images dans out_zip
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(os.path.join(tmp_dir, "model.keras"))
        with open(os.path.join(tmp_dir, "classes.txt"), "w") as f:
            f.write("\n".join(classes) + "\n")
        shutil.copytree(images_dir, os.path.join(tmp_dir, "images"))
        archive_base, _ = os.path.splitext(out_zip)
        shutil.make_archive(archive_base, "zip", tmp_dir)
    print(f"Saved bundle to '{out_zip}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a leaf disease classifier (Leaffliction Part 4).",
    )
    parser.add_argument("src", help="Directory containing class subfolders.")
    parser.add_argument("-o", "--output", default="learnings.zip",
                        help="Output .zip file (model + augmented images).")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of images kept for validation.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.src):
        print(f"Error: '{args.src}' is not a directory", file=sys.stderr)
        sys.exit(1)
    if not 0 < args.val_ratio < 1:
        print("Error: --val-ratio must be between 0 and 1", file=sys.stderr)
        sys.exit(1)
    if args.epochs < 1:
        print("Error: --epochs must be at least 1", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        print(f"Error: output directory '{out_dir}' does not exist",
              file=sys.stderr)
        sys.exit(1)

    try:
        run(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def run(args):
    paths, labels = load_dataset(args.src)
    classes = sorted(set(labels))
    print(f"Found {len(paths)} image(s) across {len(classes)} class(es)")
    check_dataset(labels, args.val_ratio)

    train_paths, train_labels, val_paths, val_labels = split_train_val(
        paths, labels, val_ratio=args.val_ratio,
    )
    print(f"Train: {len(train_paths)} image(s), "
          f"Val: {len(val_paths)} image(s)")

    with tempfile.TemporaryDirectory() as images_dir:
        train_dir = os.path.join(images_dir, "train")
        val_dir = os.path.join(images_dir, "validation")
        augment_training_set(train_paths, train_labels, train_dir)
        copy_images(val_paths, val_labels, val_dir)
        segment_directory(train_dir)
        segment_directory(val_dir)

        train_data = load_split(train_dir, classes, shuffle=True)
        val_data = load_split(val_dir, classes, shuffle=False)

        model = build_model(num_classes=len(classes))
        model.summary()
        train_model(model, train_data, val_data, epochs=args.epochs)
        save_bundle(model, classes, images_dir, args.output)


if __name__ == "__main__":
    main()
