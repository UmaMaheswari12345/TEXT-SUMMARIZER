
from flask import Flask, request, jsonify, render_template # type: ignore
from summary import extractive_summarization

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    num_sentences = int(data.get('num_sentences', 2))
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    summary = extractive_summarization(text, num_sentences)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)