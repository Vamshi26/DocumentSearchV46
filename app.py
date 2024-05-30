from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize the tokenizer and model for summarization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model_summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

documents = []
vectors = []

executor = ThreadPoolExecutor(max_workers=4)

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        return content
    except Exception as e:
        return str(e)



def add_document(text):
    vector = model.encode(text)
    documents.append(text)
    vectors.append(vector)

def search_context(query, top_k=5):
    if not vectors:
        return []

    query_vector = model.encode(query)
    distances = np.linalg.norm(np.array(vectors) - query_vector, axis=1)
    results = np.argsort(distances)[:top_k]
    return [(documents[idx], distances[idx]) for idx in results]

def summarize_text(text, max_chunk_length=1024):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', max_length=max_chunk_length, truncation=True)
    
    # If the text is too long, split it
    input_ids = inputs['input_ids'][0]
    total_length = len(input_ids)
    if total_length <= max_chunk_length:
        summary_ids = model_summarizer.generate(input_ids.unsqueeze(0), max_length=150, min_length=30, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    # Split input_ids into chunks of max_chunk_length
    chunks = [input_ids[i:i + max_chunk_length] for i in range(0, total_length, max_chunk_length)]
    summaries = []
    
    for chunk in chunks:
        summary_ids = model_summarizer.generate(chunk.unsqueeze(0), max_length=150, min_length=30, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Combine summaries into a final summary
    final_summary = ' '.join(summaries)
    return final_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath)
        add_document(text)

        return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/answer', methods=['POST'])
def answer():
    data = request.get_json()
    question = data.get('question', '')
    contexts = search_context(question)
    
    if not contexts:
        return jsonify({"summary": "No relevant context found. Please upload documents first."})

    # Concatenate contexts for summarization
    concatenated_contexts = " ".join([ctx[0] for ctx in contexts])
    
    # Asynchronous summarization
    future = executor.submit(summarize_text, concatenated_contexts)
    summary = future.result()
    
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
