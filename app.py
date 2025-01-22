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

    print("\n📥 Raw Survey Data Received:")
    print(survey_data)

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


@app.route('/paper-author', methods=['GET'])
def get_paper_author():
    """
    Returns the author's contact email (16th column) for a given paper title, 
    but only searches among the top three rows of the CSV.
    Example: GET /paper-author?title=Optimizing%20Coffee%20Harvests%20in%20Biodiverse%20Areas
    """
    paper_title = request.args.get('title', '').strip()
    if not paper_title:
        return jsonify({'error': 'No paper title provided.'}), 400

    try:
        file_path = Path('data') / 'data_with_answers.csv'
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        with file_path.open(mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)

            next(csv_reader)  # Skip header row
            for i, row in enumerate(csv_reader):
                # Limit to top 3 rows
                if i >= 3:
                    break

                # row[1] = paper title (2nd column), row[15] = author email (16th column)
                if len(row) > 15 and row[1].strip().lower() == paper_title.lower():
                    email = row[15].strip()
                    return jsonify({'email': email}), 200

        # If we didn't find a match in the top three rows:
        return jsonify({'error': f'No author found for paper title "{paper_title}" in the top three rows.'}), 404

    except Exception as e:
        print(f"Error reading file for author contact: {e}")
        return jsonify({'error': 'Failed to retrieve author information'}), 500

if __name__ == '__main__':
    app.run(debug=True)