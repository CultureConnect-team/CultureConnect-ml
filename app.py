from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

preprocessor = load("preprocessor.joblib")
model = load("CBF_model.joblib")
df_full = pd.read_json("dataset_updated.json")

def rekomendasi_tempat(preferensi_kategori_list, deskripsi_user="", rating_user=4.5, top_k=10):
    vectors = []
    for kategori in preferensi_kategori_list:
        input_df = pd.DataFrame([{
            "category": kategori,
            "description": deskripsi_user,
            "rating": rating_user
        }])
        vector = preprocessor.transform(input_df).toarray()
        vectors.append(vector)
    avg_vector = np.mean(vectors, axis=0)
    distances, indices = model.kneighbors(avg_vector, n_neighbors=top_k)
    rekomendasi = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df_full.iloc[idx]
        rekomendasi.append({
            "name": row["name"],
            "category": row["category"],
            "location": row["location"],
            "rating": row["rating"],
            "description": row["description"],
            "image": row["image"],
            "similarity": round(1 - dist, 4)
        })
    return rekomendasi

@app.route("/api/rekomendasi", methods=["POST"])
def api_rekomendasi():
    data = request.get_json()
    kategori = data.get("kategori", [])
    deskripsi = data.get("deskripsi", "")
    rating = data.get("rating", 4.5)
    hasil = rekomendasi_tempat(kategori, deskripsi, rating)
    return jsonify(hasil)

if __name__ == "__main__":
    app.run()
