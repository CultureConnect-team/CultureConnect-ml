import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Gabungkan kategori dan deskripsi jadi satu kolom
df["features"] = df["category"] + " " + df["description"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["features"])

# Hitung Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi berdasarkan wisata yang dipilih
def get_recommendations(place_name, top_n=3):
    idx = df[df["name"] == place_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_places = [df.iloc[i[0]]["name"] for i in sim_scores]
    return recommended_places

# Contoh penggunaan
if __name__ == "__main__":
    print(get_recommendations("Beach Paradise"))
