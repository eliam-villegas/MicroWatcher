from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yaml

# Cargar configuración desde el archivo YAML
with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Conectar con la base de datos
db_uri = config['database']['uri']
db_name = config['database']['dbname']
collection_name = config['database']['collection']

# Establecer conexión con MongoDB y obtener la colección
client = MongoClient(db_uri)
db = client[db_name]
collection = db[collection_name]

# Obtener los datos de MongoDB
# Filtrar y convertir los timestamps a datetime
detections = list(collection.find({}, {"_id": 0, "timestamp": 1}))
timestamps = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') for d in detections]

# Crear un DataFrame con los timestamps
df = pd.DataFrame({'timestamp': timestamps})
df['count'] = 1  # Añadir una columna de conteo

# Agrupar por intervalos de tiempo (por ejemplo, cada 15 minutos)
df = df.set_index('timestamp').resample('15T').sum().reset_index()

# Rellenar los intervalos vacíos con 0s (en caso de que no haya detecciones en ese intervalo)
df = df.set_index('timestamp').asfreq('15T', fill_value=0).reset_index()

# Calcular la hora y minuto de cada intervalo para crear una columna de tiempo
df['time_of_day'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0

# Preparar los datos para la regresión lineal
X = df[['time_of_day']].values  # Feature: tiempo del día
y = df['count'].values          # Target: número de micros detectadas

# Ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Crear predicciones para las próximas horas (por ejemplo, cada 15 minutos durante 1 hora)
next_hours = 4  # Predicción para las próximas 4 horas
future_times = np.arange(df['time_of_day'].max(), df['time_of_day'].max() + next_hours, 0.25)
future_times = future_times.reshape(-1, 1)
predictions = model.predict(future_times)

# Crear un DataFrame para almacenar las predicciones
future_df = pd.DataFrame({'time_of_day': future_times.flatten(), 'predicted_count': predictions})

# Graficar los resultados
plt.figure(figsize=(14, 7))
plt.plot(df['time_of_day'], df['count'], label='Detecciones reales', marker='o')
plt.plot(future_df['time_of_day'], future_df['predicted_count'], label='Predicción', linestyle='--', color='orange')
plt.xlabel('Hora del día')
plt.ylabel('Número de micros detectadas')
plt.title('Predicción de frecuencia de micros en las próximas horas')
plt.legend()
plt.grid()
plt.show()
