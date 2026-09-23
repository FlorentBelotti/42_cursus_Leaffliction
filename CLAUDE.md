# Leaffliction — contexte projet

Projet École 42 / RNCP 7 Architect Data. Classification d'images de
feuilles par reconnaissance de maladie. Quatre parties au sujet ;
**les parties 1 (analyse) et 2 (augmentation) sont implémentées ici**,
les parties 3 (transformation d'image) et 4 (classification) restent à
faire.

## Stack

Python 3.10, Pillow, NumPy, matplotlib. `flake8` en norme, **79
colonnes strict** (`setup.cfg`). Aucune dépendance à OpenCV ou plantCV
pour l'instant : les parties 3 et 4 en ajouteront.

## Arborescence

```
.
├── Distribution.py                       # entrypoint partie 1
├── Augmentation.py                       # entrypoint partie 2
├── balance_dataset.py                    # entrypoint équilibrage
├── requirements.txt
├── setup.cfg                             # config flake8
└── leaffliction/
    ├── dataset_analysis/                 # PARTIE 1
    │   ├── image_file_collector.py
    │   ├── class_distribution.py
    │   ├── plant_type_label_extractor.py
    │   ├── categorical_color_palette.py
    │   ├── distribution_chart_renderer.py
    │   ├── distribution_summary_printer.py
    │   └── dataset_distribution_pipeline.py
    ├── image_augmentation/               # PARTIE 2 — briques
    │   ├── augmentation_fill_color.py
    │   ├── base_augmentation.py
    │   ├── flip_augmentation.py
    │   ├── rotate_augmentation.py
    │   ├── skew_augmentation.py
    │   ├── shear_augmentation.py
    │   ├── crop_augmentation.py
    │   ├── distortion_augmentation.py
    │   ├── augmentation_registry.py
    │   ├── augmented_image_writer.py
    │   ├── augmentation_preview_renderer.py
    │   └── single_image_pipeline.py
    ├── dataset_balancing/                # PARTIE 2 — équilibrage
    │   ├── class_balancing_plan.py
    │   ├── balancing_plan_builder.py
    │   ├── augmented_dataset_copier.py
    │   ├── generation_step.py
    │   ├── dataset_balancer.py
    │   ├── balancing_report_printer.py
    │   └── dataset_balancing_pipeline.py
    ├── cli/
    │   ├── distribution_argument_parser.py
    │   ├── augmentation_execution_mode.py
    │   ├── augmentation_argument_parser.py
    │   └── balance_dataset_argument_parser.py
    └── utils/
        ├── supported_image_extensions.py
        ├── filesystem_path_validator.py
        ├── image_file_loader.py
        ├── perspective_coefficient_solver.py
        ├── chart_style.py
        ├── figure_display.py
        └── error_reporting.py
```

## Classes clés et responsabilités

| Classe / module | Rôle unique |
|---|---|
| `ImageFileCollector` | Parcourt récursivement un répertoire et groupe les images par classe. La classe d'une image = nom du répertoire qui la contient directement. |
| `ClassDistribution` | Vue en lecture seule des effectifs par classe. Expose les noms triés, les comptes alignés, le max, le min, le total. |
| `DistributionChartRenderer` | Construit la figure matplotlib (camembert + histogramme) et l'écrit sur disque. |
| `categorical_color_palette` | Palette catégorielle fixe de 8 teintes, assignée par index alphabétique → couleur stable d'un graphe à l'autre et d'un run à l'autre. |
| `BaseAugmentation` (ABC) | Contrat : `augmentation_name`, `apply_to_image`, `build_variant_with_index`. Aucune transformation n'utilise de tirage aléatoire. |
| `FlipAugmentation` | Miroir horizontal ou vertical (`ImageOps`). |
| `RotateAugmentation` | Rotation bicubique, cadre conservé, coins remplis en blanc. |
| `SkewAugmentation` | Transformation **perspective** (trapèze) — les parallèles cessent de l'être. |
| `ShearAugmentation` | Transformation **affine** (parallélogramme) recentrée — les parallèles le restent. C'est ce qui la distingue visuellement de Skew. |
| `CropAugmentation` | Recadrage ancré puis redimensionnement à la taille d'origine (cadre homogène pour la partie 4). |
| `DistortionAugmentation` | Ondulation sinusoïdale par remapping d'indices NumPy, bords répliqués par clamp. |
| `AugmentationRegistry` | Ordre canonique unique des 6 transformations (ordre d'affichage **et** ordre du round-robin). |
| `AugmentedImageWriter` | Nommage `<stem>_<Type><ext>`, plus `_v<n>` au-delà du premier variant. |
| `AugmentationPreviewRenderer` | Planche d'une ligne : original + 6 variants labellisés. |
| `BalancingPlanBuilder` | Cible = effectif de la classe majoritaire. Retourne un `ClassBalancingPlan` par classe. |
| `AugmentedDatasetCopier` | Duplique l'arbre source avant augmentation ; refuse une destination existante sans `--force`. |
| `DatasetBalancer` | Round-robin déterministe (image, transformation, variant) jusqu'à atteindre la cible. |

## Contraintes techniques spécifiques

- **Déterminisme total.** Aucun `random`. Le tri des chemins dans
  `ImageFileCollector` + le round-robin de `DatasetBalancer` garantissent
  que deux exécutions produisent un `augmented_directory` bit-à-bit
  identique. C'est nécessaire pour que le `signature.txt` (sha1 du zip du
  dataset) reste valable.
- **Cadre conservé.** Toutes les transformations rendent une image de la
  taille d'origine. La partie 4 attend un tenseur de forme fixe.
- **Remplissage blanc** (`AUGMENTATION_FILL_COLOR`) là où la géométrie ne
  laisse pas de pixel — cohérent avec le fond clair du dataset et avec la
  segmentation de la partie 3.
- **Le dataset original n'est jamais modifié** par l'équilibrage : la
  copie est faite d'abord, l'augmentation écrit dans la copie.
- **Nommage des fichiers augmentés** conforme au `ls` du sujet :
  `_Flip`, `_Rotate`, `_Skew`, `_Shear`, `_Crop`, `_Distortion`.
- **Le dataset ne doit jamais être commité** (interdit par le sujet).
  `.gitignore` couvre `augmented_directory/`, `images/`, `*.zip`.
- **Headless.** `figure_display` détecte l'absence de backend interactif
  et n'échoue pas ; toute figure peut être écrite sur disque à la place.

## Round-robin de l'équilibrage (détail)

Pour une classe de `N` images et `6` transformations, à l'étape `i` :

```
image        = images_triées[i % N]
transformation = séquence[(i // N) % 6]
variant      = i // (N * 6)
```

Conséquence : toutes les images de la classe passent par `Flip` avant
qu'une seule ne passe par `Rotate`. La diversité ajoutée est étalée, et
aucun doublon n'apparaît avant `N * 6` générations.

## Usage

```bash
./Distribution.py ./Apple
./Distribution.py ./Apple --output charts/apple.png --no-display

./Augmentation.py 'Apple/apple_healthy/image (1).JPG'
./Augmentation.py ./Apple --destination augmented_directory
./Augmentation.py ./Apple --mode dataset --force
./balance_dataset.py ./Apple -d augmented_directory

python3 -m flake8 .
```

`Augmentation.py` déduit son mode du chemin reçu (fichier → mode image,
répertoire → mode dataset) ; `--mode {auto,image,dataset}` force le
choix. `balance_dataset.py` est le même pipeline sous un nom explicite —
une seule implémentation, deux points d'entrée.

## État d'avancement

| Partie | État |
|---|---|
| 1 — Analyse du dataset (`Distribution.py`) | **fait**, flake8 propre, testé |
| 2 — Data augmentation (`Augmentation.py`, `balance_dataset.py`) | **fait**, flake8 propre, testé, déterminisme vérifié |
| 3 — Image Transformation (`Transformation.py`) | pas commencé |
| 4 — Classification (`train.py`, `predict.py`) | pas commencé |
| `signature.txt` (sha1 du zip du dataset) | pas commencé — dépend de la partie 4 |

Ce qui n'est **pas** encore fait et sera nécessaire : `Transformation.py`
(gaussian blur, mask, roi objects, analyze object, pseudolandmarks,
histogramme de couleur — plantCV recommandé), la séparation
train/validation, `train.py`, `predict.py`, et le `signature.txt`.

---

# MON STANDARD DE CODE (à respecter strictement)

## Nommage
- Les noms de fonctions et de variables doivent être longs et
  transparents sur leur rôle
  → Préférer `calculate_monthly_revenue_per_user()` à `calc_rev()`
- Casse cohérente : snake_case en Python, camelCase en TypeScript
  → Pas de mélange dans un même fichier
- Noms de fichiers et de dossiers en snake_case, transparents sur leur
  contenu
  → `user_authentication_handler.py` plutôt que `auth.py`
  → `data_pipeline/ingestion/raw_file_parser.py`

## Découpage du code
- Toute logique identifiable doit être encapsulée dans sa propre fonction
- Toute sous-logique dans une fonction doit être extraite en
  sous-fonction
- Un ensemble cohérent de fonctions liées au même process → une classe
- Une classe = un rôle, un fichier = une responsabilité claire (SRP)
- Privilégier beaucoup de petites fonctions plutôt que peu de grosses

## Structure fichiers et dossiers
- Découper en plusieurs fichiers dès qu'un fichier grossit ou mélange
  des responsabilités
- Organiser en dossiers reflétant l'architecture logique du projet
  (ex. /ingestion, /transformation, /storage, /api, /utils...)
- Chaque dossier a un rôle unique et lisible exprimé dans son nom

## Style
- Code chronologique et lisible : l'ordre du code reflète l'ordre
  d'exécution
- Zéro ternaire : toujours des if/else explicites
- Pas d'optimisation au détriment de la lisibilité
- Docstring obligatoire au niveau des fonctions ; commentaires inline
  uniquement quand la logique n'est pas évidente
