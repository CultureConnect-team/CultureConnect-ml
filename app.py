from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

try:
    preprocessor = load("preprocessor.joblib")
    model = load("CBF_model.joblib")
    df_full = pd.read_json("dataset_updated.json")
    valid_categories = df_full['category'].unique().tolist()
except Exception as e:
    print(f"Error loading resources: {str(e)}")
    traceback.print_exc()
    valid_categories = []

def get_recommendations(category_list, user_description="", user_rating=4.5, top_k=100):

    if len(category_list) < 3:
        raise ValueError("Minimal 3 kategori harus dipilih.")

    category_list = [cat for cat in category_list if cat in valid_categories]
    if not category_list:
        return []

    vectors = []
    for category in category_list:
        try:
            input_df = pd.DataFrame([{
                "category": category,
                "description": user_description,
                "rating": user_rating
            }])
            vector = preprocessor.transform(input_df).toarray()
            vectors.append(vector)
        except Exception as e:
            print(f"Error processing category {category}: {str(e)}")
            continue

    if not vectors:
        return []

    avg_vector = np.mean(vectors, axis=0)
    distances, indices = model.kneighbors(avg_vector, n_neighbors=top_k)
    
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df_full.iloc[idx]
        recommendations.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "location": str(row["location"]),
            "rating": float(row["rating"]),
            "description": str(row["description"]),
            "image": str(row["image"]),
            "similarity": round(1 - float(dist), 4)
        })

    
    return recommendations

@app.route("/", methods=["GET", "POST"])
def index():
    form_data = {
        "kategori": "",
        "deskripsi": "",
        "rating": "4.5"
    }
    
    context = {
        "input_data": form_data,
        "valid_categories": valid_categories,
        "rekomendasi": None,
        "error": None
    }

    if request.method == "POST":
        # Get form data
        form_data["kategori"] = request.form.get("kategori", "")
        form_data["deskripsi"] = request.form.get("deskripsi", "")
        form_data["rating"] = request.form.get("rating", "4.5")
        
        try:
            # Process form inputs
            categories = [k.strip() for k in form_data["kategori"].split(",") if k.strip()]
            description = form_data["deskripsi"]
            rating = float(form_data["rating"])
            
            # Get recommendations
            recommendations = get_recommendations(categories, description, rating)
            
            # Update context
            context.update({
                "rekomendasi": recommendations,
                "input_data": form_data
            })
            
            # Add error message if no results
            if not recommendations and categories:
                error_msg = "Kategori tidak dikenali."
                if valid_categories:
                    error_msg += f" Coba dengan kategori seperti: {', '.join(valid_categories[:3])}..."
                context["error"] = error_msg
                
        except ValueError:
            context["error"] = "Format rating tidak valid"
        except Exception as e:
            context["error"] = f"Terjadi kesalahan: {str(e)}"
    
    return render_template("index.html", **context)

@app.route("/api/rekomendasi", methods=["POST"])
def api_recommendations():
    """API endpoint for recommendations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Data tidak valid"}), 400
            
        # Process input data
        categories = data.get("kategori", [])
        if isinstance(categories, str):
            categories = [k.strip() for k in categories.split(",") if k.strip()]
            
        description = data.get("deskripsi", "")
        
        rating = float(data.get("rating", 4.5))

        
        # Get recommendations
        results = get_recommendations(categories, description, rating)
        return jsonify({"rekomendasi": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)