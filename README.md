# Leaffliction

Image classification by disease recognition on leaves — École 42.

Parts implemented in this repository: **analysis of the data set** and
**data augmentation**.

## Install

```bash
pip install -r requirements.txt
```

## Part 1 — Analysis of the data set

```bash
./Distribution.py ./Apple
./Distribution.py ./Apple --output charts/apple.png --no-display
```

Walks the subdirectories, counts the images of every class, prints a
textual summary and plots a pie chart and a bar chart. The figure is
titled after the directory given on the command line. The class of an
image is the name of the directory that directly contains it, so the
program works on `./Apple` as well as on the data set root.

## Part 2 — Data augmentation

Six augmentations: `Flip`, `Rotate`, `Skew`, `Shear`, `Crop`,
`Distortion`.

### One image

```bash
./Augmentation.py 'Apple/apple_healthy/image (1).JPG'
```

Displays the original next to its six variants, and writes them beside
the original as `image (1)_Flip.JPG`, `image (1)_Rotate.JPG`, and so on.

### A whole data set

```bash
./Augmentation.py ./Apple --destination augmented_directory
./balance_dataset.py ./Apple -d augmented_directory
```

Copies the data set, then generates augmented images inside the copy
until every class holds as many images as the most populated one. The
original data set is left untouched.

Every transformation is deterministic: two runs produce the same
`augmented_directory`, so its sha1 signature is stable.

Use `-h` on any of the three programs for the full option list.

## Norm

```bash
python3 -m flake8 .
```
