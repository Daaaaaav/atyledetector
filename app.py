from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('clothing_style_model.h5')  


class_labels = [
    "athletic wear", "Casual", "Formal", "Other", "Streetwear"
]

advice_dict = {
    "athletic wear": "Perfect for workouts or active days! Pair with comfortable sneakers and a light jacket for outdoor activities.",
    "Casual": "A relaxed style that's perfect for everyday wear. Consider pairing with jeans or shorts and a simple t-shirt.",
    "Formal": "For formal events, consider wearing a tailored suit or dress. Pair it with dress shoes or heels for a polished look.",
    "Other": "Looks unique! Consider accessorizing with jewelry or a statement piece to make your outfit stand out.",
    "Streetwear": "Stay trendy with streetwear! Hoodies, graphic tees, and sneakers make a bold statement."
}

def predict_clothing_style(image_file):
    img = Image.open(image_file).convert("RGB")  
    img = img.resize((150, 150)) 
    
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)  
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  
    predicted_class = class_labels[predicted_class_idx]  
    
    confidence = predictions[0][predicted_class_idx] 
    return predicted_class, confidence

def get_clothing_advice(style_class):
    return advice_dict.get(style_class, "Sorry, I don't have specific advice for this style.")

@app.route('/')
def home():
    return render_template('chat.html') 

@app.route('/get_text_response', methods=['POST'])
def get_text_response():
    user_message = request.json.get('message') 
    if 'clothing' in user_message.lower() or 'fashion' in user_message.lower():
        response = get_clothing_advice(user_message)
    else:
        response = "Sorry, I can only provide clothing advice at the moment."

    return jsonify({"response": response}) 

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files: 
        return jsonify({"error": "No image file uploaded"}), 400
    
    image = request.files['image']
    if image.filename == '':  
        return jsonify({"error": "No selected file"}), 400
    
    try:
        predicted_class, confidence = predict_clothing_style(image)
        advice = get_clothing_advice(predicted_class)
        response = {
            "style": predicted_class, 
            "confidence": f"{confidence * 100:.2f}%",
            "advice": advice  
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

if __name__ == '__main__':
    app.run(debug=True)  
