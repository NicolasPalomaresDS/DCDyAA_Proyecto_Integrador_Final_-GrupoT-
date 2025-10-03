# --- prepare_dataset_v2: crea dataset listo para YOLOv8-seg y compatibilidad con el notebook original ---
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import random
import cv2

def prepare_dataset(root='.', patients_dir='Patients_CT', hem_csv='hemorrhage_diagnosis.csv',
                       out_dir='dataset_yolo_seg', new_size=(512,512), train_ratio=0.8,
                       include_no_mask_as_negatives=True, random_seed=42):
    """
    Prepara dataset de imágenes y máscaras para YOLOv8-seg.
    
    Parámetros:
        root (str): Directorio raíz.
        patients_dir (str): Directorio de los datos y tomografías.
        hem_csv (str): CSV con información de cortes, anotaciones y diagnóstico.
        out_dir (str): Carpeta de salida para dataset YOLO.
        new_size (tuple): Tamaño de resize para imágenes y máscaras.
        train_ratio (float): Proporción de train/val.
        include_no_mask_as_negatives (bool): Si True, genera máscaras vacías cuando no existe máscara.
        random_seed (int): Semilla para reproducibilidad.
    
    Retorna:
        manifest_df (DataFrame): Manifest con paths de imágenes y máscaras.
        stats (dict): Estadísticas del procesamiento.
    """
    root = Path(root)
    patients_dir = root / patients_dir
    hem_csv = root / hem_csv
    out = root / out_dir
    out_images_train = out / "images" / "train"
    out_images_val = out / "images" / "val"
    out_images_no_mask = out / "images" / "no_mask"
    out_masks_train = out / "masks" / "train"
    out_masks_val = out / "masks" / "val"
    for p in [out_images_train, out_images_val, out_images_no_mask, out_masks_train, out_masks_val]:
        p.mkdir(parents=True, exist_ok=True)
    random.seed(random_seed)

    if not patients_dir.exists():
        raise FileNotFoundError(f"No encontré {patients_dir} desde {root}")

    if not hem_csv.exists():
        raise FileNotFoundError(f"No encontré {hem_csv} desde {root}")

    df = pd.read_csv(hem_csv)
    arr = df.values
    # build map patient -> list of slices from the CSV
    by_patient = {}
    for row in arr:
        pid = int(row[0])
        slice_no = int(row[1])
        by_patient.setdefault(pid, []).append(slice_no)

    patient_ids = sorted(by_patient.keys())
    random.shuffle(patient_ids)
    cutoff = int(len(patient_ids) * train_ratio)
    train_ids = set(patient_ids[:cutoff])

    manifest_rows = []
    counter = 0
    stats = {"processed":0, "with_mask":0, "no_mask":0, "missing_image":0}
    for pid in patient_ids:
        split = "train" if pid in train_ids else "val"
        for slice_no in sorted(by_patient[pid]):
            src_img = patients_dir / f"{pid:03d}" / "brain" / f"{slice_no}.jpg"
            if not src_img.exists():
                stats["missing_image"] += 1
                continue
            img = Image.open(src_img).convert("L")
            img_r = img.resize(new_size, resample=Image.BILINEAR)
            basename = f"{pid:03d}_{slice_no}"
            out_img_path = (out / "images" / split / f"{basename}.png")
            img_r.save(out_img_path, format="PNG")

            src_mask = patients_dir / f"{pid:03d}" / "brain" / f"{slice_no}_HGE_Seg.jpg"
            if src_mask.exists():
                m = Image.open(src_mask).convert("L")
                m_r = m.resize(new_size, resample=Image.NEAREST)
                m_arr = np.array(m_r)
                m_bin = np.where(m_arr > 0, 255, 0).astype(np.uint8)
                out_mask_path = (out / "masks" / split / f"{basename}.png")
                Image.fromarray(m_bin, mode="L").save(out_mask_path, format="PNG")
                stats["with_mask"] += 1
                mask_exists = True
            else:
                mask_exists = False
                stats["no_mask"] += 1
                if include_no_mask_as_negatives:
                    out_mask_path = (out / "masks" / split / f"{basename}.png")
                    Image.fromarray(np.zeros((new_size[1], new_size[0]), dtype=np.uint8), mode="L").save(out_mask_path, format="PNG")
                else:
                    out_mask_path = None
                    shutil.copy2(out_img_path, out_images_no_mask / f"{basename}.png")

            manifest_rows.append({
                "index": counter,
                "patient_id": pid,
                "slice": slice_no,
                "split": split,
                "image_path": str(out_img_path),
                "mask_exists": bool(mask_exists),
                "mask_path": str(out_mask_path) if out_mask_path else ""
            })
            counter += 1
            stats["processed"] += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out / "manifest.csv", index=False)
    return manifest_df, stats

# ====================================================================================

def convert_masks_to_yolo(dataset_dir, splits=("train", "val"), class_id=0):
    """
    Convierte máscaras binarias en anotaciones YOLO formato .txt.
    
    Parámetros:
        dataset_dir (str | Path): Carpeta base del dataset con 'masks'.
        splits (tuple): Conjuntos a procesar (por defecto 'train' y 'val').
        class_id (int): ID de la clase para YOLO.
    
    Retorna:
        None (genera archivos .txt en labels/).
    """
    
    dataset_dir = Path(dataset_dir)
    for split in splits:
        masks_dir = dataset_dir / "masks" / split
        labels_dir = dataset_dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for mask_path in masks_dir.glob("*.png"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            # Umbralizar (por seguridad)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            label_path = labels_dir / (mask_path.stem + ".txt")
            with open(label_path, "w") as f:
                for contour in contours:
                    if len(contour) < 3:
                        continue  # descartar contornos inválidos
                    # Aproximar polígono
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Normalizar coordenadas
                    norm_coords = []
                    for point in approx:
                        x, y = point[0]
                        norm_x = x / w
                        norm_y = y / h
                        norm_coords.append(f"{norm_x:.6f} {norm_y:.6f}")
                    
                    # Escribir en formato YOLO: class_id + coords
                    line = f"{class_id} " + " ".join(norm_coords)
                    f.write(line + "\n")
                    
                    