from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load
import numpy as np

app = Flask(__name__)

# Load the ML model
model = load_model('model.h5')

# Load the tokenizers
with open('kan_tokenizer.pkl', 'rb') as f:
    tokenizer_kan = load(f)

with open('tulu_tokenizer.pkl', 'rb') as f:
    tokenizer_tulu = load(f)

# Translation route
@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        input_text = request.json['text']  # Assuming JSON input
        
        # Preprocess the input text
        input_sequence = tokenizer_kan.texts_to_sequences([input_text])
        
        # Calculate the length of the input sequence dynamically
        input_length = len(input_sequence[0])
        
        # Pad the input sequence based on its length
        input_sequence = pad_sequences(input_sequence, maxlen=input_length, padding='post')
        
        # Perform translation using the ML model
        translated_probabilities = model.predict(input_sequence)
        
        # Convert probabilities to letters
        translated_indices = [np.argmax(probabilities) for probabilities in translated_probabilities[0]]
        translated_text = ' '.join(tokenizer_tulu.index_word[index] for index in translated_indices if index != 0)
        
        return jsonify({'translated_text': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
