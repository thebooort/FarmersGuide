from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_handler import generate_response

app = Flask(__name__)
CORS(app)

# In-memory storage for survey data
survey_context = {}

@app.route('/survey', methods=['POST'])
def receive_survey():
    """
    Receives survey data and stores it in memory.
    """
    global survey_context
    survey_data = request.get_json()  # Parse incoming JSON data
    if not survey_data:
        return jsonify({"error": "No survey data received"}), 400

    # Store the survey data
    survey_context = {
        "Location": survey_data.get("Location", "N/A"),
        "Crop": survey_data.get("Crop", "N/A"),
        "Ecosystem": survey_data.get("Ecosystem", "N/A"),
        "Agriculture": survey_data.get("Agriculture", "N/A")
    }

    print("🔍 Survey Data Stored:", survey_context)
    return jsonify({"message": "Survey data stored successfully"}), 200

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    """
    Handles chat requests, incorporating survey context.
    """
    global survey_context
    data = request.get_json()  # Ensure JSON parsing
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Log survey context being used
    print("\n🔍 Survey Context Used:")
    print(survey_context)

    try:
        # Get response from RAG handler
        ai_response = generate_response(user_query=user_message, survey_context=survey_context)

        # Ensure response is always JSON
        return jsonify(ai_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
