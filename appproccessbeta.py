import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# Leer el archivo CSV en trozos
chunk_size = 100000  # ajusta según tus necesidades
data_chunks = pd.read_csv('rating.csv', chunksize=chunk_size)

# Inicializar el modelo NearestNeighbors fuera del bucle si es posible
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)

# Procesar cada trozo
for chunk in data_chunks:
    # Seleccionar las columnas que se utilizarán para el cálculo de vecinos más cercanos
    X = chunk[['userId', 'movieId', 'rating']]

    # Ajustar el modelo NearestNeighbors a los datos
    model.fit(X)

    # Paralelizar el cálculo de vecinos
    results = Parallel(n_jobs=-1)(delayed(model.kneighbors)(X.iloc[i:i+1]) for i in range(len(X)))

    # Imprimir los resultados
    for i, (distances, indices) in enumerate(results):
        print(f"Vecinos más cercanos para el punto {i + chunk_size}:")
        for j in range(1, len(indices[0])):
            neighbor_index = indices[0][j]
            distance = distances[0][j]
            neighbor_data = chunk.iloc[neighbor_index]
            print(f"  Vecino {j}: Usuario {neighbor_data['userId']}, Película {neighbor_data['movieId']}, Calificación {neighbor_data['rating']}, Distancia {distance}")
        print("\n")
