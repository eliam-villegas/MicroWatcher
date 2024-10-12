# src/db_handler.py

from pymongo import MongoClient

class DatabaseHandler:
    def __init__(self, uri, db_name, collection_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_detection(self, detection_data):
        try:
            self.collection.insert_one(detection_data)
            print("Detección guardada correctamente en la base de datos.")
        except Exception as e:
            print(f"Error al guardar detección en la base de datos: {e}")
