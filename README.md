# CultureConnect Machine Learning API 🤖  

Welcome to the **CultureConnect Machine Learning API**! 🚀  
This repository contains the ML model that provides recommendations based on user preferences and cultural data. It powers AI-driven features in the **CultureConnect** platform.  

---

## 📌 Features  
✅ **Recommendation System** – Suggests destinations based on user preferences.  
✅ **Dataset Integration** – Uses structured data for training and inference.  
✅ **Flask API** – Serves predictions through an HTTP interface.  
✅ **Virtual Environment** – Isolated dependencies using `venv`.  

---

## 🚀 Technologies Used  
- **Python** 🐍 – Primary language for ML development.  
- **Flask** 🌐 – Lightweight web framework to serve ML models.  
- **Pandas** 📊 – Data manipulation and analysis.  
- **NumPy** 🔢 – Efficient numerical computing.  
- **scikit-learn** 🤖 – Machine learning algorithms.  

---

## 📂 Folder Structure  
```
ml/
│-- venv/                 # Virtual environment (dependencies isolated)
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── pyvenv.cfg
│
│-- .gitignore             # Ignore unnecessary files (venv, cache, etc.)
│-- app.py                 # Flask application to serve the ML model
│-- dataset.csv            # Dataset used for training and recommendations
│-- LICENSE                # Project license
│-- README.md              # Documentation (this file)
│-- recommendations.py     # Core recommendation engine logic
│-- requirements.txt       # Required dependencies for the ML model
```

---

## 📥 How to Set Up and Run the Project  
### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/CultureConnect-team/CultureConnect-ML.git
```

### 2️⃣ Create a Virtual Environment  
Navigate to the project directory and create a virtual environment:  
```sh
cd CultureConnect-ML
python -m venv venv
```

### 3️⃣ Activate the Virtual Environment  
- **Windows**  
  ```sh
  venv\Scripts\activate
  ```
- **macOS/Linux**  
  ```sh
  source venv/bin/activate
  ```

### 4️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 5️⃣ Run the Flask API  
```sh
python app.py
```
The API will be available at **http://127.0.0.1:5000** or at **http://127.0.0.1:5001**.

---

## 🛠 API Endpoints  
### 🔹 **Recommendations**  
| METHOD | ENDPOINT            | DESCRIPTION |
|--------|--------------------|-------------|
| POST   | `/recommend`       | Get personalized destination recommendations |

#### 📌 Example Request (JSON)  
```json
{
  "user_preferences": ["nature", "culture", "historical"]
}
```

#### 📌 Example Response (JSON)  
```json
{
  "recommendations": [
    {"destination": "Bali", "score": 0.92},
    {"destination": "Yogyakarta", "score": 0.87}
  ]
}
```

---

## 🏗 How It Works  
1. **Dataset (`dataset.csv`)** – Contains information about various tourist destinations, categorized by features such as culture, adventure, history, etc.  
2. **ML Model (`recommendations.py`)** – Uses **scikit-learn** to process user input and generate recommendations.  
3. **Flask API (`app.py`)** – Serves the ML model through an HTTP API.  

---

## 🤝 Contributing  
We welcome contributions! Follow these steps:  
1. **Fork** the repository.  
2. **Create a new branch** (`feature/your-feature-name`).  
3. **Make changes** and commit with a clear message.  
4. **Push** your branch and create a **Pull Request (PR)**.  

> Please follow Python best practices and formatting (PEP 8).  

---

## 🛠 Troubleshooting  
If you encounter issues:  
- Ensure **Python** is installed (`python --version`).  
- Activate the virtual environment before running the API.  
- If dependencies fail, try:  
  ```sh
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Restart the API with `python app.py`.  

---

## 📌 License  
This project is licensed under the **MIT License** – see the [`LICENSE`](LICENSE) file for details.  

---

🚀 **Happy coding!** 🎯
