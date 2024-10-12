# src/detection.py

import torch

class MicrobusDetection:
    def __init__(self, model_path, confidence_threshold=0.6):
        # Establecer el dispositivo (GPU si está disponible, de lo contrario CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")

        # Cargar el modelo YOLOv5 entrenado con el archivo .pt proporcionado
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        print("Modelo cargado correctamente")

        # Configurar el umbral de confianza para las detecciones
        self.model.conf = confidence_threshold
        self.model.iou = 0.45  # Umbral de IoU para la supresión no máxima
        print(f"Umbral de confianza configurado en: {confidence_threshold}")

    def detect_microbus(self, frame):
        # Realizar la detección en el frame (la imagen se pasa en formato RGB)
        results = self.model(frame[:, :, ::-1])  # Convertir BGR (de OpenCV) a RGB

        # Obtener los resultados como un DataFrame de pandas
        detections_df = results.pandas().xyxy[0]

        # Filtrar las detecciones para encontrar sólo microbuses (asegúrate de ajustar esto según el índice o el nombre de tu clase)
        microbus_detections = []
        for _, row in detections_df.iterrows():
            # Verificar si la detección corresponde a la clase deseada, como "Autobus Naranja Lincosur"
            if row['name'] == "Autobus Naranja Lincosur":  # Reemplaza con el nombre correcto de tu clase
                # Obtener las coordenadas del bounding box y la confianza
                box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                score = row['confidence']
                microbus_detections.append((box, score))

        return microbus_detections


