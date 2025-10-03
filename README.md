# Proyecto Final Integrador: Clasificación y Detección de Hemorragias Intracraneales
Este repositorio contiene el código, datos y recursos utilizados en el Proyecto Final Integrador de la Diplomatura en Ciencia de Datos y Análisis Avanazdo del Centro de E-Learning
de la Universidad Tecnológica Nacional. El objetivo es predecir y clasificar mediante técnicas de Machine Learning diferentes tipos de hemorragias intracraneales a partir de datos
de diagnóstico y demográficos de determinados pacientes, y detectar y segmentar la lesión en un dataset de tomografías computarizadas utilizando Redes Neuronales Convolucionales
(YOLO).

### Estructura del directorio
--
* `notebook_proyecto_final.ipunb`: Notebook de Jupyter con la carga de los datos, análisis, preprocesamiento, ingeniería de variables, modelado (LightGBM, YOLOv8), evaluación de las métricas, explicación de variables con SHAP y pruebas de desempeño.
*  `data_yaml_yolov8_seg.yaml`: Archivo de configuración del modelo YOLOv8.
* `data/`: Directorio con todos los datos necesarios descargados de Kaggle, incluyendo datasets en formato CSV, dataset de tomografías (base y transformadas para el modelo), y archivos de ayuda y licensias proporcionadas por el autor.
* `gui/`: Directorio con la interfaz gráfica de prueba del modelo YOLO en formato script de Python.
* `runs/`: Pruebas de validación, resultados de entrenamiento, gráficos y pesos del modelo final YOLOv8s entrenado.
* `utils/`: Directorio con los scripts y funciones utilizados para el modelado de los algoritmos (`ML_models.py`), balanceo de datos e imágenes con augmentations (`balance_dataset.py`), preparación y separación de las imágenes (`split_data.py`) y prueba de desempeño para el modelo YOLO (`test_YOLO.py`).
* `yolo_models/`: Directorio con el modelo YOLOv8s base.
* `README.md`: Archivo de lectura del repositorio actual.

### Requisitos
---
* Python 3.11 o superior.
* Librerías: `pandas`, `matplotlib`, `seaborn`, `numpy`, `opencv-python`, `scikit-learn`, `tensorflow`, `ultralytics`, `kaggle`
Pasos para la instalación y ejecución de la notebook:
1. Clonar o descargar el repositorio.
2. Ejecutar la notebook en su entorno de preferencia (Jupyter, Colab, Visual Studio Code, etc.).
3. Instalar dependencias necesarias: `pip install pandas matplotlib seaborn numpy opencv-python scikit-learn tensorflow ultralytics kaggle` (celda incluída en la notebook)
4. Configurar archivo `.kaggle` en el sistema (con cuenta en Kaggle) para la correcta descarga de los datos.
La notebook descarga automáticamente los datos si se realizaron los pasos correspondientes.
No se recomienda ejecutar la celda que entrena el modelo YOLO, ya que dependiendo la capacidad física de procesamiento puede demorar mucho tiempo (!)

### Citaciones
---
* Datos: Kaggle - Brain CT Images with Intracranial Hemorrhage Masks
* Enlace: [https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images]
