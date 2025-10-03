import random
from pathlib import Path
import shutil
import albumentations as A
import cv2

def balance(dataset_dir, split="train", ratio=2):
    """
    Balancea el dataset creando un subconjunto con una proporción definida 
    de negativos por cada positivo.

    Parámetros:
        dataset_dir (str | Path): Ruta a la carpeta raíz del dataset (ej. 'dataset_yolo_seg').
        split (str): Subconjunto a procesar: 'train' o 'val'.
        ratio (int): Cantidad de negativos a mantener por cada positivo (ej. 2 = 1 positivo : 2 negativos).

    Retorna:
        None: Modifica/crea directorios y archivos en disco (images/labels balanceados).
    """
    
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    
    pos_imgs, neg_imgs = [], []
    
    for label_file in labels_dir.glob("*.txt"):
        if label_file.stat().st_size > 0:
            pos_imgs.append(label_file.stem)
        else:
            neg_imgs.append(label_file.stem)
    
    print(f"{split}: {len(pos_imgs)} positivos, {len(neg_imgs)} negativos")
    
    # Elegir subset de negativos
    keep_neg = random.sample(neg_imgs, min(len(neg_imgs), len(pos_imgs)*ratio))
    keep = set(pos_imgs + keep_neg)
    
    # Crear carpetas balanceadas
    out_images = dataset_dir / f"images_{split}_balanced"
    out_labels = dataset_dir / f"labels_{split}_balanced"
    out_images.mkdir(exist_ok=True, parents=True)
    out_labels.mkdir(exist_ok=True, parents=True)
    
    for stem in keep:
        shutil.copy(images_dir / f"{stem}.png", out_images / f"{stem}.png")
        shutil.copy(labels_dir / f"{stem}.txt", out_labels / f"{stem}.txt")
    
    print(f"Guardado en {out_images}, {out_labels}")
    
# ==================================================================================

def oversample_positives(dataset_dir, split="train", factor=2):
    """
    Duplica o replica imágenes positivas (con hemorragia) para aumentar su representación 
    en el dataset balanceado.

    Parámetros:
        dataset_dir (str | Path): Ruta a la carpeta raíz del dataset (ej. 'dataset_yolo_seg').
        split (str): Subconjunto a procesar: 'train' o 'val'.
        factor (int): Número de copias adicionales por cada imagen positiva.

    Retorna:
        None: Genera copias de las imágenes/labels positivas en disco.
    """
    
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / f"images_{split}_balanced"
    labels_dir = dataset_dir / f"labels_{split}_balanced"
    
    pos_imgs = [p for p in labels_dir.glob("*.txt") if p.stat().st_size > 0]
    
    for label_file in pos_imgs:
        stem = label_file.stem
        for i in range(factor-1):
            # copiar con nuevo nombre
            new_stem = f"{stem}_aug{i}"
            shutil.copy(images_dir / f"{stem}.png", images_dir / f"{new_stem}.png")
            shutil.copy(label_file, labels_dir / f"{new_stem}.txt")
            
# ==================================================================================

def move_files(src, dst):
    """
    Mueve todos los archivos de un directorio a otro y elimina el directorio original.

    Parámetros:
        src (str | Path): Carpeta origen de los archivos.
        dst (str | Path): Carpeta destino donde se moverán los archivos.

    Retorna:
        None: Realiza la operación de movimiento en disco; no devuelve valor.
    """
    
    if src.exists():
        for file in src.glob("*"):
            shutil.move(str(file), dst)
        print(f"✅ Movidos {len(list(dst.glob('*')))} archivos de {src.name} a {dst}")
        # Borrar la carpeta vieja
        shutil.rmtree(src)
        
def reorganize_directories(dir):
    """
    Reorganiza las carpetas del dataset para que coincidan con la estructura esperada por YOLO
    (por ejemplo: images/train_balanced, labels/train_balanced, images/val_balanced, ...).

    Parámetros:
        dir (str | Path): Carpeta raíz del dataset (ej. 'dataset_yolo_seg').

    Retorna:
        None: Ajusta la estructura de carpetas y crea/borra directorios según sea necesario.
    """
    
    base_dir = Path(dir)

    # Carpetas actuales
    old_images_train = base_dir / "images_train_balanced"
    old_labels_train = base_dir / "labels_train_balanced"
    old_images_val   = base_dir / "images_val_balanced"
    old_labels_val   = base_dir / "labels_val_balanced"

    # Carpetas destino (estructura que YOLO espera)
    new_images_train = base_dir / "images/train_balanced"
    new_labels_train = base_dir / "labels/train_balanced"
    new_images_val   = base_dir / "images/val_balanced"
    new_labels_val   = base_dir / "labels/val_balanced"

    # Crear carpetas destino si no existen
    for d in [new_images_train, new_labels_train, new_images_val, new_labels_val]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Mover todo
    move_files(old_images_train, new_images_train)
    move_files(old_labels_train, new_labels_train)
    move_files(old_images_val, new_images_val)
    move_files(old_labels_val, new_labels_val)

    print("\nEstructura reorganizada correctamente ✅")
    
# ==================================================================================
    
def augment_positives(images_dir, labels_dir, n_aug=3):
    """
    Aplica técnicas de data augmentation con Albumentations solo a las imágenes positivas
    (con hemorragia) y guarda versiones aumentadas junto a las originales.

    Parámetros:
        images_dir (str | Path): Carpeta con imágenes originales (positivas).
        labels_dir (str | Path): Carpeta con labels en formato YOLO correspondientes.
        n_aug (int): Número de augmentaciones a generar por cada imagen positiva.

    Retorna:
        None: Guarda las imágenes aumentadas y sus labels correspondientes en disco.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Pipeline de augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(p=0.2, alpha=120, sigma=120*0.05, alpha_affine=120*0.03)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Recorrer labels
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        
        img_name = label_file.stem + ".png"
        img_path = images_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bboxes, class_labels = [], []
        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            bboxes.append([x, y, bw, bh])
            class_labels.append(int(cls))

        for i in range(n_aug):
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            # Guardar imagen aumentada en el mismo directorio
            out_img_path = images_dir / f"{label_file.stem}_aug{i}.png"
            cv2.imwrite(str(out_img_path), aug_img)

            # Guardar labels en el mismo directorio
            out_label_path = labels_dir / f"{label_file.stem}_aug{i}.txt"
            with open(out_label_path, "w") as f:
                for cls, bbox in zip(aug_labels, aug_bboxes):
                    x, y, bw, bh = bbox
                    f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Augmentations agregadas en {images_dir} y {labels_dir}")

