from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Determine the directory of the current script (api.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root directory to sys.path (one level up from api.py)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import functions from main.py (assuming it's in the project root)
from main import youtube_audio_to_text, load_summarizer, summarize_long_text

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Load the summarizer *once* when the app starts
summarizer = load_summarizer()

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML template

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url')

        if not youtube_url:
            return jsonify({'error': 'Missing YouTube URL'}), 400

        result = youtube_audio_to_text(youtube_url)
        if result:
            transcription, _ = result  # Get transcription from the returned tuple
            summary = summarize_long_text(summarizer, transcription)
            return jsonify({'summary': summary})
        else:
            return jsonify({'error': 'Failed to transcribe or summarize video'}), 500

    except Exception as e:
        # Log the exception in a real application and return a generic error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)