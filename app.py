from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")
df["features"] = df["category"] + " " + df["description"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(place_name, top_n=3):
    if place_name not in df["name"].values:
        return []
    idx = df[df["name"] == place_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_places = [df.iloc[i[0]]["name"] for i in sim_scores]
    return recommended_places

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    place_name = data.get("place_name")
    recommendations = get_recommendations(place_name)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
