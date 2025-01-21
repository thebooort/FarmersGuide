from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama import chat
from pathlib import Path
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    data = request.json
    user_message = data.get('message', '')

    try:
        response = chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': user_message}]
        )
        ai_response = response.message.content
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/survey', methods=['POST'])
def receive_survey():
    survey_data = request.json

    # Log the survey data for debugging purposes
    print("Survey Data Received:")
    print(f"Location: {survey_data.get('Location', 'N/A')}")
    print(f"Crop: {survey_data.get('Crop', 'N/A')}")
    print(f"Ecosystem: {survey_data.get('Ecosystem', 'N/A')}")
    print(f"Agriculture: {survey_data.get('Agriculture', 'N/A')}")

    return jsonify({'message': 'Survey data received successfully!', 'data': survey_data}), 200

@app.route('/pregenerated-questions', methods=['GET'])
def get_pregenerated_questions():
    questions = []
    try:
        # Use Path to ensure compatibility across operating systems
        file_path = Path('data') / 'data_with_answers.csv'
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        with file_path.open(mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Get the header row
            questions = header[9:13]  # Columns 10-13 (0-indexed)
    except Exception as e:
        print(f"Error reading file: {e}")
        return jsonify({'error': 'Failed to retrieve questions'}), 500

    return jsonify({'questions': questions}), 200

@app.route('/papers', methods=['GET'])
def get_papers():
    papers = []
    try:
        # Use Path to ensure compatibility across operating systems
        file_path = Path('data') / 'data_with_answers.csv'
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        with file_path.open(mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row if it exists
            for i, row in enumerate(csv_reader):
                if i >= 3:  # Limit to top 3 rows
                    break
                papers.append(row[1])  # Assuming the second column contains the paper titles
    except Exception as e:
        print(f"Error reading file: {e}")
        return jsonify({'error': 'Failed to retrieve papers'}), 500

    return jsonify({'papers': papers}), 200

if __name__ == '__main__':
    app.run(debug=True)