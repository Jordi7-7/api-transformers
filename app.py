from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time


# Inicializar Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Ruta donde guardaste el modelo
output_dir = "./transformers_imdb_bert_model"

# Cargar el modelo y el tokenizador
model = AutoModelForSequenceClassification.from_pretrained(output_dir)  # Cargar modelo desde el archivo `model.safetensors`
tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Cargar el tokenizador desde `tokenizer.json`

# Configurar el dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Enviar el modelo al dispositivo

# Ruta de prueba
@app.route('/test', methods=['GET'])
def hello_world():
    texto = "Me encantó este producto, es excelente"
    return jsonify({'message': texto})


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Endpoint para realizar predicciones con una imagen enviada como multipart/form-data.
    """ 
    if 'review' not in request.form:
        return jsonify({"error": "No se proporcionó el campo 'review'"}), 400

    try:

        # Leer la reseña desde form-data
        review = request.form['review']
        if not isinstance(review, str) or not review.strip():
            return jsonify({"error": "La reseña debe ser un texto válido"}), 400

        # Tokenizar el texto de prueba
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Enviar datos al mismo dispositivo que el modelo

        # Cambiar el modelo a modo de evaluación
        model.eval()  

        # Hacer la predicción
        # Medir el tiempo de inferencia
        start_time = time.time()  # Inicio del temporizador
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()  # Fin del temporizador

        inference_time = end_time - start_time  # Tiempo de inferencia en segundos

        # Obtener la clase predicha
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()

        # Mapear resultado a positivo/negativo
        sentiment = "positivo" if predicted_class == 1 else "negativo"

        # Formatear y devolver la respuesta
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
