from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load
import numpy as np

app = Flask(__name__)

# Load the ML model
model = load_model('api/model.h5')

# Load the tokenizers
with open('/kan_tokenizer.pkl', 'rb') as f:
    tokenizer_kan = load(f)

with open('api/tulu_tokenizer.pkl', 'rb') as f:
    tokenizer_tulu = load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST','GET'])
def translate_text():
    try:
        input_text = request.json['text']  # Assuming JSON input
        
        print("Received input text:", input_text)  # Add this line for debugging
        
        # Preprocess the input text
        input_sequence = tokenizer_kan.texts_to_sequences([input_text])
        
        # Calculate the length of the input sequence dynamically
        # input_length = len(input_sequence[0])
        
        # Pad the input sequence based on its length
        input_sequence = pad_sequences(input_sequence, maxlen=24, padding='post')
        
        # Perform translation using the ML model
        translated_probabilities = model.predict(input_sequence)
        
        # Convert probabilities to letters
        translated_indices = [np.argmax(probabilities) for probabilities in translated_probabilities[0]]
        translated_text = ' '.join(tokenizer_tulu.index_word[index] for index in translated_indices if index != 0)
        
        return jsonify({'translated_text': translated_text}), 200
    except Exception as e:
        print("Error:", e)  # Add this line for debugging
        return jsonify({'error': str(e)}), 500

    

if __name__ == '__main__':
    app.run(debug=True)
