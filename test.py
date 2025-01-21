import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ruta donde guardaste el modelo
output_dir = "./transformers_imdb_bert_model"

# Cargar el modelo y el tokenizador
model = AutoModelForSequenceClassification.from_pretrained(output_dir)  # Cargar modelo desde el archivo `model.safetensors`
tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Cargar el tokenizador desde `tokenizer.json`

# Configurar el dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Enviar el modelo al dispositivo

# Texto de prueba
text = "This movie was absolutely awesome! I loved every moment of it."

# Tokenizar el texto de prueba
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
inputs = {key: value.to(device) for key, value in inputs.items()}  # Enviar datos al mismo dispositivo que el modelo

# Hacer la predicción
model.eval()  # Cambiar el modelo a modo de evaluación
with torch.no_grad():
    outputs = model(**inputs)

print(model)

# Obtener la clase predicha
predicted_class = torch.argmax(outputs.logits, dim=-1).item()

# Mostrar el resultado
print(f"Predicción: {'Positivo' if predicted_class == 1 else 'Negativo'}")
