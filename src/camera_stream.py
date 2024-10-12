# src/camera_stream.py

import cv2
import time
import yaml
from detection import MicrobusDetection
from db_handler import DatabaseHandler

def main():
    # Cargar configuración desde el archivo YAML
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Inicializar el detector de microbuses
    model_path = config['model']['path']
    confidence_threshold = config['model']['confidence_threshold']
    detector = MicrobusDetection(model_path, confidence_threshold)

    # Conectar con la base de datos
    db_uri = config['database']['uri']
    db_name = config['database']['dbname']
    collection_name = config['database']['collection']
    db_handler = DatabaseHandler(db_uri, db_name, collection_name)

    # Conectar con la cámara
    camera_url = config['camera']['url']

    if isinstance(camera_url, int):
        cap = cv2.VideoCapture(camera_url)
    else:
        try:
            cap = cv2.VideoCapture(int(camera_url))
        except ValueError:
            cap = cv2.VideoCapture(camera_url)

    # Configurar el ancho y alto del frame
    frame_width = config['camera']['width']
    frame_height = config['camera']['height']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    print("Cámara conectada. Iniciando detección...")

    # Inicializar el registro de detecciones
    last_detection_time = {}
    detection_delay = 5  # Tiempo en segundos para permitir la detección
    distance_threshold = 50  # Distancia en píxeles para considerar que es la misma microbus

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: No se pudo leer el frame de la cámara")
            break

        # Realizar detección en el frame
        detections = detector.detect_microbus(frame)
        print(f"Detecciones: {detections}")

        # Guardar detecciones en la base de datos
        current_time = time.time()  # Obtener el tiempo actual
        for box, score in detections:
            box_key = tuple(box)  # Usar la caja como clave

            # Calcular el centro del bounding box
            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            # Verificar si esta detección ya fue registrada recientemente
            detected = False
            for key in last_detection_time.keys():
                last_center = last_detection_time[key]['center']
                last_time = last_detection_time[key]['time']

                # Calcular la distancia entre los centros
                distance = ((box_center[0] - last_center[0]) ** 2 + (box_center[1] - last_center[1]) ** 2) ** 0.5

                # Verificar si la distancia es menor que el umbral y el tiempo es menor que el intervalo permitido
                if distance < distance_threshold and (current_time - last_time < detection_delay):
                    detected = True
                    break

            if not detected:
                # Si no se detectó en el intervalo de tiempo, guardamos la detección
                detection_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'box': [float(coord) for coord in box],
                    'score': float(score)
                }
                db_handler.insert_detection(detection_data)

                # Actualizar el registro de detección
                last_detection_time[box_key] = {'time': current_time, 'center': box_center}  # Guardar tiempo y centro

        # Mostrar el frame con detecciones (Opcional)
        for (xmin, ymin, xmax, ymax), score in detections:
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)

        # Mostrar el video
        cv2.imshow('Microbus Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
