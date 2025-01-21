from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama import chat

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


if __name__ == '__main__':
    app.run(debug=True)
