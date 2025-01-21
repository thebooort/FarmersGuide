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

if __name__ == '__main__':
    app.run(debug=True)
