from pprint import pprint
from ragbackend import initialize_vectorstore, create_prompt_templates, create_workflow
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

flaskApp = Flask(__name__)
CORS(flaskApp) 

UPLOAD_FOLDER = 'Path to your folder where u want to save pdf'  # Define your upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@flaskApp.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        file.save(file_path)
        print(f"File saved to {file_path}")  # Debug log
    except Exception as e:
        print(f"Error saving file: {e}")  # Error log
        return jsonify({'error': 'Error saving file'}), 500

    # Simulate a URL for the uploaded PDF
    pdf_url = f'/uploads/{file.filename}'

    print(f"PDF URL: {pdf_url}")  # Debug log

    global vectorstore, retriever, question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader, workflow
    vectorstore  = initialize_vectorstore(file_path)
    question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader = create_prompt_templates()
    workflow = create_workflow(vectorstore, question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader)

    return jsonify({'pdfUrl': pdf_url})

@flaskApp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@flaskApp.route('/chat', methods=['POST'])
def chat():
    global counter
    counter = 0

    data = request.json
    pdf_url = data.get('pdfUrl')
    prompt = data.get('message')

    result = ""

    inputs = {"question": prompt}
    for output in workflow.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])
    result = value["generation"]

    return jsonify({'reply': result})

if __name__ == '__main__':
    flaskApp.run(debug=True)
