from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model & preprocessor
preprocessor = load("preprocessor.joblib")
model = load("CBF_model.joblib")

# Load data asli (buat nampilin hasil)
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

@app.route("/", methods=["GET", "POST"])
def index():
    rekomendasi = []
    if request.method == "POST":
        kategori_input = request.form["kategori"]
        kategori_list = [k.strip() for k in kategori_input.split(",")]
        deskripsi = request.form["deskripsi"]
        rating = float(request.form["rating"])
        rekomendasi = rekomendasi_tempat(kategori_list, deskripsi, rating)
    return render_template("index.html", rekomendasi=rekomendasi)

if __name__ == "__main__":
    app.run(debug=True)
