"""
Interfaz mínima para probar un modelo YOLO (segmentación).
- Modelo fijo (definido en código)
- Abrir imagen -> predicción automática
- Mostrar Original + Predicción
- Guardar máscara opcional
"""

import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# Configuración
MODEL_PATH = "../runs/segment/train_yolo_final/weights/best.pt"
CONF = 0.25
IMGSZ = 640

# Cargar modelo fijo
model = YOLO(MODEL_PATH)

photo_refs = []  # Para evitar que tkinter borre imágenes
current_img_path = None  # Guardar última imagen abierta


# utilidades
def load_image_cv2(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pil_from_cv2_gray(img_gray, target_w=None, target_h=None):
    if target_w is not None and target_h is not None:
        img_gray = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)


def overlay_mask_on_gray(img_gray, mask, alpha=0.45):
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    red_layer = np.zeros_like(rgb)
    red_layer[:, :, 2] = 255
    bin_mask = (mask > 0).astype(np.uint8)
    overlay = rgb.copy().astype(np.float32)
    overlay[bin_mask == 1] = (1.0 - alpha) * overlay[bin_mask == 1] + alpha * red_layer[bin_mask == 1]
    return np.clip(overlay, 0, 255).astype(np.uint8)


# lógica principal
def open_image():
    global current_img_path
    path = filedialog.askopenfilename(title="Abrir imagen", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not path:
        return
    current_img_path = path
    img_gray = load_image_cv2(path)
    if img_gray is None:
        status_label.config(text="⚠️ No se pudo leer la imagen.")
        return
    show_original(img_gray)
    predict_and_show(path)


def show_original(img_gray):
    h, w = img_gray.shape
    max_dim = 512
    scale = min(max_dim / max(h, w), 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    pil_img = pil_from_cv2_gray(img_gray, display_w, display_h)
    imgtk = ImageTk.PhotoImage(pil_img)
    photo_refs.clear()
    photo_refs.append(imgtk)
    left_label.config(image=imgtk)


def predict_and_show(img_path):
    status_label.config(text="⏳ Ejecutando predicción...")
    root.update_idletasks()

    res = model.predict(source=str(img_path), conf=CONF, imgsz=IMGSZ, save=False)[0]
    orig = load_image_cv2(img_path)
    H, W = orig.shape

    pred_mask = None
    if res.masks is not None and res.masks.data is not None:
        mask_arr = res.masks.data.cpu().numpy()
        combined = (mask_arr.sum(axis=0) > 0).astype(np.uint8) * 255
        pred_mask = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST)

    if pred_mask is not None:
        overlay_img_rgb = overlay_mask_on_gray(orig, pred_mask, alpha=0.45)
    else:
        try:
            plotted = res.plot()
            overlay_img_rgb = plotted[..., ::-1] if plotted.shape[2] == 3 else plotted
        except Exception:
            overlay_img_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    max_dim = 512
    scale = min(max_dim / max(H, W), 1.0)
    display_w, display_h = int(W * scale), int(H * scale)
    pil_overlay = Image.fromarray(overlay_img_rgb).resize((display_w, display_h), Image.Resampling.LANCZOS)
    imgtk2 = ImageTk.PhotoImage(pil_overlay)
    photo_refs.append(imgtk2)
    right_label.config(image=imgtk2)

    status_label.config(text=f"✅ Predicción lista (conf={CONF}, imgsz={IMGSZ})")


def save_mask():
    if not current_img_path:
        status_label.config(text="⚠️ No hay imagen cargada.")
        return
    res = model.predict(source=current_img_path, conf=CONF, imgsz=IMGSZ, save=False)[0]
    orig = load_image_cv2(current_img_path)
    H, W = orig.shape
    mask = None
    if res.masks is not None and res.masks.data is not None:
        mask_arr = res.masks.data.cpu().numpy()
        combined = (mask_arr.sum(axis=0) > 0).astype(np.uint8) * 255
        mask = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask is None:
        status_label.config(text="⚠️ No se pudo generar máscara.")
        return
    out_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
    if out_path:
        cv2.imwrite(out_path, mask)
        status_label.config(text=f"💾 Máscara guardada en {out_path}")


# Construcción UI
root = tk.Tk()
root.title("YOLO - Demo simple (segmentación)")

frm = ttk.Frame(root, padding=8)
frm.grid(row=0, column=0, sticky="nsew")

ctrl_frame = ttk.Frame(frm)
ctrl_frame.grid(row=0, column=0, sticky="w", pady=4)

btn_open = ttk.Button(ctrl_frame, text="Abrir imagen", command=open_image)
btn_open.grid(row=0, column=0, padx=4)

btn_save = ttk.Button(ctrl_frame, text="Guardar máscara...", command=save_mask)
btn_save.grid(row=0, column=1, padx=8)

img_frame = ttk.Frame(frm)
img_frame.grid(row=1, column=0, pady=8)

left_label = ttk.Label(img_frame)
left_label.grid(row=0, column=0, padx=4)
right_label = ttk.Label(img_frame)
right_label.grid(row=0, column=1, padx=4)

status_label = ttk.Label(frm, text="Modelo cargado. Abrir una imagen para probar.", foreground="blue")
status_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.mainloop()
