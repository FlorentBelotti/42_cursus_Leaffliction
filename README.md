# Leaffliction

Classification de maladies de feuilles par computer vision
(sujet 42 - Leaffliction).

## Installation

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` crée un virtualenv dans `/goinfre/$USER/leaffliction_venv`
(modifiable via `VENV_DIR`), le lie en `.venv` et installe les dépendances
listées dans `requirements.txt` (numpy, matplotlib, plantcv, flake8,
tensorflow, scikit-learn).

## Partie 1 : Distribution

`Distribution.py` compte les images de chaque classe (nom du dossier qui
les contient) et affiche un camembert et un histogramme par type de plante.

```bash
./Distribution.py leaves/images/Apple        # une figure "apple"
./Distribution.py leaves/images              # une figure par plante
./Distribution.py leaves/images -o charts/dist.png --no-display
```

Le type de plante est le dossier intermédiaire (`images/Apple/apple_healthy`)
ou, pour un dossier plat (`images/Apple_healthy`), le préfixe du nom de
classe avant le premier `_`.

## Partie 2 : Augmentation

Image unique -> 6 variantes (Flip, Rotate, Skew, Shear, Crop, Distortion)
enregistrées à côté de l'original sous `image (1)_Flip.JPG` etc., et
affichées côte à côte :

```bash
./Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"
```

Dossier -> copie équilibrée du data set dans `augmented_directory` :

```bash
./Augmentation.py leaves/images -d augmented_directory
```

## Partie 3 : Transformation

`Transformation.py` applique 6 transformations d'image à une feuille : Blur,
Mask, Roi, Analyze, Landmarks, Histogram (voir le code pour le détail de
chacune).

Image unique -> affichage à l'écran :

```bash
./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"
```

Dossier -> sauvegarde dans un dossier de destination :

```bash
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory
```

Un flag (`-blur`, `-mask`, `-roi`, `-analyze`, `-landmarks`, `-histogram`)
restreint la génération à une seule transformation :

```bash
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory -mask
```

## Partie 4 : Classification

`train.py` sépare les images en train (80 %) et validation (20 %), équilibre
le seul train set avec les augmentations de la partie 2, détoure les
feuilles, entraîne un CNN et écrit `learnings.zip` (modèle + classes +
images modifiées, dossiers `images/train` et `images/validation`).

```bash
./train.py leaves/images                     # -> learnings.zip
./train.py leaves/images --epochs 15 -o learnings.zip
```

`predict.py` affiche l'image originale, l'image transformée et la classe
prédite. Donné un dossier, il mesure la précision (preuve des > 90 % sur la
validation) :

```bash
./predict.py "leaves/images/Apple_healthy/image (1).JPG"
unzip -q learnings.zip 'images/validation/*' -d bundle
./predict.py bundle/images/validation
```

## Signature (obligatoire pour le rendu)

`signature.txt` à la racine doit contenir le sha1 du `.zip` contenant le
data set et l'entraînement. Après le dernier `train.py` :

```bash
./make_signature.sh learnings.zip
git add signature.txt
```

Ne pas relancer `train.py` ensuite : le zip, donc le hash, changerait.
Le `.zip` lui-même ne doit pas être commité (il est dans `.gitignore`).

## Norme

```bash
flake8 .
```
