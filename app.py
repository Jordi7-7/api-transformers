from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

# Inicializar Flask
app = Flask(__name__)

# Ruta donde guardaste el modelo
output_dir = "transformers_imdb_bert_model"

# Cargar el modelo y el tokenizador
model = AutoModelForSequenceClassification.from_pretrained(output_dir)  # Cargar modelo desde el archivo `model.safetensors`
tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Cargar el tokenizador desde `tokenizer.json`

# Configurar el dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Enviar el modelo al dispositivo

# Ruta para servir el archivo index.html
@app.route('/')
def home():
    # Renderiza index.html desde el directorio actual
    return send_from_directory(os.getcwd(), "index.html")

# Ruta de predicción
@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'review' not in request.form:
        return jsonify({"error": "No se proporcionó el campo 'review'"}), 400

    try:
        review = request.form['review']
        if not isinstance(review, str) or not review.strip():
            return jsonify({"error": "La reseña debe ser un texto válido"}), 400

        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Enviar datos al mismo dispositivo que el modelo

        model.eval()
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()

        inference_time = end_time - start_time

        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        sentiment = "positivo" if predicted_class == 1 else "negativo"

        return jsonify({
            "review": review,
            "sentiment": sentiment,
            "predicted_class": predicted_class,
            "inference_time": f"{inference_time:.4f} segundos"
        })

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error al procesar la reseña: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
