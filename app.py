from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    target_user_id = user_id
    model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)

    # Leer los datos del archivo CSV en un objeto DataFrame de Pandas
    data = pd.read_csv('rating.csv')

    # Filtrar datos para el usuario específico
    user_data = data[data['userId'] == user_id]

    # Seleccionar las columnas que se utilizarán para el cálculo de vecinos más cercanos
    X_user = user_data[['userId', 'movieId', 'rating']]

    # Ajustar el modelo NearestNeighbors a los datos del usuario específico
    model.fit(X_user)

    # Paralelizar el cálculo de vecinos para el usuario específico
    results = Parallel(n_jobs=-1)(delayed(model.kneighbors)(X_user.iloc[i:i+1]) for i in range(len(X_user)))

    recommendations = []

    # Recopilar recomendaciones
    for i, (distances, indices) in enumerate(results):
        movie_recommendations = []
        for j in range(1, len(indices[0])):
            neighbor_index = indices[0][j]
            distance = distances[0][j]
            recommended_movie_data = data.iloc[neighbor_index]
            movie_recommendations.append({
                'movie_id': recommended_movie_data['movieId'],
                'rating': recommended_movie_data['rating'],
                'distance': distance
            })
        recommendations.append({
            'user_id': user_id,
            'movie_id': user_data.iloc[i]['movieId'],
            'recommendations': movie_recommendations
        })

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)


