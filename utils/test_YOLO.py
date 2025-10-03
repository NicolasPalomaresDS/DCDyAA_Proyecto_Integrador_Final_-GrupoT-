import matplotlib.pyplot as plt
import numpy as np
import cv2

def test_images(model, val_img_dir, val_lbl_dir, N):
    """
    Visualiza predicciones YOLO vs ground truth en imágenes de validación.
    
    Parámetros:
        model: Modelo YOLO cargado (ultralytics).
        val_img_dir (Path): Directorio con imágenes de validación.
        val_lbl_dir (Path): Directorio con anotaciones YOLO de validación.
        N (int): Número de imágenes a mostrar.
    
    Retorna:
        None (muestra gráficos matplotlib con Original, Predicción, Ground Truth).
    """
    
    img_files = list(val_img_dir.glob("*.png"))

    for img_path in img_files[:N]:
        # Imágen original
        orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

        # Predicción
        res = model.predict(source=str(img_path), conf=0.25, imgsz=640, save=False)[0]
        pred_mask = None
        if res.masks is not None and res.masks.data is not None:
            mask_array = res.masks.data.cpu().numpy()
            pred_mask = (mask_array.sum(axis=0) > 0).astype(np.uint8) * 255
            pred_mask = cv2.resize(pred_mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ground truth
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        gt_mask = None
        if lbl_path.exists():
            gt_mask = np.zeros_like(orig, dtype=np.uint8)
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 5:  
                        cls = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        poly = np.array(coords).reshape(-1, 2) * [orig.shape[1], orig.shape[0]]
                        poly = poly.astype(np.int32)
                        cv2.fillPoly(gt_mask, [poly], 255)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(orig_rgb, cmap="gray")
        axs[0].set_title("Imagen Original")

        if pred_mask is not None:
            axs[1].imshow(orig_rgb, cmap="gray")
            axs[1].imshow(pred_mask, cmap="Reds", alpha=0.5)
            axs[1].set_title("Predicción YOLO")
        else:
            axs[1].imshow(orig_rgb, cmap="gray")
            axs[1].set_title("Sin predicción")

        if gt_mask is not None:
            axs[2].imshow(orig_rgb, cmap="gray")
            axs[2].imshow(gt_mask, cmap="Blues", alpha=0.5)
            axs[2].set_title("Ground Truth")
        else:
            axs[2].imshow(orig_rgb, cmap="gray")
            axs[2].set_title("Sin ground truth")

        for ax in axs:
            ax.axis("off")
        plt.show()