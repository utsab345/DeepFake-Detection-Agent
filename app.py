"""
Flask Web Application for Deepfake Detection Agent System

This web app provides a beautiful, interactive interface for the multi-agent
deepfake detection system with real-time communication and stunning UI.
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agent_system import DeepfakeDetectionOrchestrator
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

# Initialize extensions
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global orchestrator instance (will be initialized when API key is available)
orchestrator = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_orchestrator():
    """Initialize the orchestrator if API key is available."""
    global orchestrator
    if orchestrator is None:
        try:
            if Config.GEMINI_API_KEY:
                orchestrator = DeepfakeDetectionOrchestrator(Config.GEMINI_API_KEY)
                logger.info("Orchestrator initialized successfully")
            else:
                logger.warning("GEMINI_API_KEY not set - LLM features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")


# Routes
@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check API and agent status."""
    return jsonify({
        'status': 'online',
        'agent_ready': orchestrator is not None,
        'gemini_configured': bool(Config.GEMINI_API_KEY),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle image upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, WEBP, GIF'}), 400
    
    try:
        # Save file with secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {
        'message': 'Connected to Deepfake Detection Agent System',
        'agent_ready': orchestrator is not None
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('analyze_image')
def handle_analyze_image(data):
    """Handle image analysis request via WebSocket."""
    try:
        filepath = data.get('filepath')
        user_query = data.get('query', '')
        
        if not filepath or not os.path.exists(filepath):
            emit('error', {'message': 'Image file not found'})
            return
        
        # Emit status update
        emit('agent_status', {
            'agent': 'Detection Agent',
            'status': 'analyzing',
            'message': 'Analyzing image for deepfakes...'
        })
        
        # Initialize orchestrator if needed
        if orchestrator is None:
            init_orchestrator()
        
        if orchestrator is None:
            # Fallback to basic detection without LLM
            from deepfake_detector import DeepfakeDetector
            detector = DeepfakeDetector()
            
            emit('agent_status', {
                'agent': 'Detection Agent',
                'status': 'running',
                'message': 'Running ML model...'
            })
            
            result = detector.predict(filepath)
            
            if 'error' in result:
                emit('error', {'message': result['error']})
                return
            
            emit('detection_result', result)
            emit('agent_status', {
                'agent': 'Detection Agent',
                'status': 'complete',
                'message': 'Detection complete'
            })
            
            # Send basic explanation (no LLM)
            explanation = f"The image was classified as '{result['label']}' with {result['confidence']:.1%} confidence. "
            if result['label'] == 'Realism':
                explanation += "This suggests the image is likely a real photograph, not AI-generated."
            else:
                explanation += "This suggests the image may be AI-generated or manipulated."
            
            emit('analysis_result', {'explanation': explanation})
        
        else:
            # Full agent workflow with LLM
            emit('agent_status', {
                'agent': 'Orchestrator',
                'status': 'coordinating',
                'message': 'Coordinating agents...'
            })
            
            # Process through orchestrator
            analysis = orchestrator.process_image(filepath, user_query)
            
            # Get the last detection result from memory
            detection_result = None
            for msg in reversed(orchestrator.memory.messages):
                if msg.role == 'system' and msg.metadata:
                    detection_result = msg.metadata
                    break
            
            if detection_result:
                emit('detection_result', detection_result)
            
            emit('agent_status', {
                'agent': 'Analysis Agent',
                'status': 'complete',
                'message': 'Analysis complete'
            })
            
            emit('analysis_result', {'explanation': analysis})
        
        logger.info(f"Analysis complete for: {filepath}")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        emit('error', {'message': f'Analysis failed: {str(e)}'})


@socketio.on('ask_question')
def handle_ask_question(data):
    """Handle follow-up question via WebSocket."""
    try:
        question = data.get('question', '')
        
        if not question:
            emit('error', {'message': 'No question provided'})
            return
        
        if orchestrator is None:
            emit('error', {'message': 'Agent system not initialized. Please configure GEMINI_API_KEY.'})
            return
        
        emit('agent_status', {
            'agent': 'Analysis Agent',
            'status': 'thinking',
            'message': 'Processing your question...'
        })
        
        answer = orchestrator.ask_question(question)
        
        emit('agent_status', {
            'agent': 'Analysis Agent',
            'status': 'complete',
            'message': 'Answer ready'
        })
        
        emit('question_answer', {
            'question': question,
            'answer': answer
        })
        
        logger.info(f"Question answered: {question[:50]}...")
    
    except Exception as e:
        logger.error(f"Question error: {e}", exc_info=True)
        emit('error', {'message': f'Failed to answer question: {str(e)}'})


@socketio.on('save_session')
def handle_save_session():
    """Handle session save request."""
    try:
        if orchestrator is None:
            emit('error', {'message': 'No active session to save'})
            return
        
        orchestrator.save_session()
        
        emit('session_saved', {
            'message': 'Session saved successfully',
            'session_id': orchestrator.memory.session_id
        })
        
        logger.info(f"Session saved: {orchestrator.memory.session_id}")
    
    except Exception as e:
        logger.error(f"Save session error: {e}")
        emit('error', {'message': f'Failed to save session: {str(e)}'})


if __name__ == '__main__':
    # Initialize orchestrator on startup
    init_orchestrator()
    
    print("\n" + "="*60)
    print("🚀 Deepfake Detection Web App Starting...")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"🤖 Agent System: {'Ready' if orchestrator else 'Limited (no API key)'}")
    print("="*60 + "\n")
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
