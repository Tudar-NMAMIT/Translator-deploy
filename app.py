from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load
import numpy as np

app = Flask(__name__)

# Load the ML model
model = load_model('model.h5')

# Load the tokenizer
with open('kan_tokenizer.pkl', 'rb') as f:
    tokenizer_kan = load(f)

with open('tulu_tokenizer.pkl', 'rb') as f:
    tokenizer_tulu = load(f)

# Translation route
@app.route('/')
def home():
    # Define translated_text to avoid Pylance warning
    translated_text = ""
    return render_template('index.html', translated_text=translated_text)

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        input_text = request.form['text']
        
        # Preprocess the input text
        input_sequence = tokenizer_kan.texts_to_sequences([input_text])
        input_sequence = pad_sequences(input_sequence, maxlen=24, padding='post')
        
        # Perform translation using the ML model
        translated_probabilities = model.predict(input_sequence)
        
        # Convert probabilities to letters
        translated_indices = [np.argmax(probabilities) for probabilities in translated_probabilities[0]]
        translated_text = ' '.join(tokenizer_tulu.index_word[index] for index in translated_indices if index != 0)
        
        return render_template('index.html', translated_text=translated_text)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
