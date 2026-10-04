#!/usr/bin/env python3
"""Leaf disease prediction (Leaffliction - Part 4).

Loads the model saved by train.py in its .zip, then:
Single image -> displays the original and the segmented image together
                with the predicted class.
Directory    -> predicts every image (label = parent directory name)
                and prints the accuracy, to check the validation set.
"""
import argparse
import os
import sys
import tempfile
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from leaffliction.leaf_segmentation import prepare_for_model
from train import IMG_SHAPE, find_images

BUNDLE_FILES = ("model.keras", "classes.txt")
BATCH_SIZE = 32
MIN_VALIDATION_IMAGES = 100  # minimum exigé par le sujet


def load_bundle(zip_path):
    # extrait le modèle et l'ordre des classes du zip créé par train.py
    with zipfile.ZipFile(zip_path) as archive, \
            tempfile.TemporaryDirectory() as tmp_dir:
        for name in BUNDLE_FILES:
            if name not in archive.namelist():
                raise ValueError(f"'{zip_path}' has no '{name}': "
                                 "is it a bundle made by train.py?")
            archive.extract(name, tmp_dir)
        model = keras.models.load_model(os.path.join(tmp_dir, "model.keras"))
        with open(os.path.join(tmp_dir, "classes.txt")) as f:
            classes = [line.strip() for line in f if line.strip()]
    # index de sortie du softmax -> classes[index]
    if model.output_shape[-1] != len(classes):
        raise ValueError(f"model predicts {model.output_shape[-1]} classes "
                         f"but classes.txt lists {len(classes)}")
    return model, classes


def load_image(path):
    # même prétraitement que segment_directory dans train.py
    bgr_img = cv2.imread(path)
    if bgr_img is None:
        raise ValueError(f"'{path}' is not a readable image")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    model_img, segmented = prepare_for_model(rgb_img)
    return rgb_img, model_img, segmented


def to_model_input(model_img):
    # redimensionnement bilinéaire, comme image_dataset_from_directory ;
    # pas de /255 ici : la couche Rescaling est dans le modèle
    return tf.image.resize(model_img, IMG_SHAPE[:2]).numpy()


def show_prediction(rgb_img, model_img, class_name, confidence):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), facecolor="black")
    for ax, img, title in ((axes[0], rgb_img, "Original"),
                           (axes[1], model_img, "Transformed")):
        ax.imshow(img)
        ax.set_title(title, color="white")
        ax.axis("off")
    fig.text(0.5, 0.12, "===   DL classification   ===",
             ha="center", color="white", fontsize=18)
    fig.text(0.5, 0.05, f"Class predicted : {class_name} "
             f"({confidence:.1%})", ha="center", color="limegreen",
             fontsize=14)
    fig.subplots_adjust(bottom=0.2)
    plt.show()


def predict_image(model, classes, path):
    rgb_img, model_img, segmented = load_image(path)
    # predict attend un lot d'images -> ajoute une dimension (1, h, w, 3)
    probas = model.predict(to_model_input(model_img)[np.newaxis],
                           verbose=0)[0]
    index = int(np.argmax(probas))
    if not segmented:
        print("Warning: no leaf detected, predicting on the raw image",
              file=sys.stderr)
    print(f"Class predicted: {classes[index]} ({probas[index]:.1%})")
    show_prediction(rgb_img, model_img, classes[index], probas[index])


def evaluate_directory(model, classes, directory):
    # label attendu = nom du dossier parent, comme dans train.py
    paths, labels = [], []
    for path, label in find_images(directory):
        paths.append(path)
        labels.append(label)
    if not paths:
        raise ValueError(f"no image found in '{directory}'")
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"unknown class folder(s): {', '.join(unknown)} "
                         f"(model knows: {', '.join(classes)})")

    scores = {name: [0, 0] for name in sorted(set(labels))}  # [ok, total]
    for start in range(0, len(paths), BATCH_SIZE):
        batch = np.stack([to_model_input(load_image(path)[1])
                          for path in paths[start:start + BATCH_SIZE]])
        predicted = np.argmax(model.predict(batch, verbose=0), axis=1)
        for label, index in zip(labels[start:start + BATCH_SIZE], predicted):
            scores[label][0] += classes[index] == label
            scores[label][1] += 1

    for name, (ok, total) in scores.items():
        print(f"{name:<30} {ok:>5}/{total:<5} {ok / total:.1%}")
    correct = sum(ok for ok, _ in scores.values())
    print(f"{'Accuracy':<30} {correct:>5}/{len(paths):<5} "
          f"{correct / len(paths):.1%}")
    if len(paths) < MIN_VALIDATION_IMAGES:
        print(f"Warning: only {len(paths)} image(s), the subject asks for "
              f"at least {MIN_VALIDATION_IMAGES}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the disease of a leaf (Leaffliction Part 4). "
                    "Give an image to display its prediction, or a "
                    "directory of class subfolders to measure accuracy.",
    )
    parser.add_argument("path", help="Image, or directory of class folders.")
    parser.add_argument("-m", "--model", default="learnings.zip",
                        help="Bundle .zip created by train.py.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.path):
        print(f"Error: '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)
    if not zipfile.is_zipfile(args.model):
        print(f"Error: '{args.model}' is not a .zip file", file=sys.stderr)
        sys.exit(1)

    try:
        model, classes = load_bundle(args.model)
        if os.path.isdir(args.path):
            evaluate_directory(model, classes, args.path)
        else:
            predict_image(model, classes, args.path)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
