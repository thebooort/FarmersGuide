from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_handler import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """Handles AI chat requests and forwards to RAG handler"""
    data = request.get_json()  # Ensure JSON parsing
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message received'}), 400

    try:
        # Get response from RAG handler
        ai_response = generate_response(user_message)
        
        # Ensure response is always JSON
        return jsonify(ai_response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)

