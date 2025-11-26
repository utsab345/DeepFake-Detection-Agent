// WebSocket connection
const socket = io();

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const agentStatus = document.getElementById('agentStatus');
const agentStatusText = document.getElementById('agentStatusText');

// State
let uploadedFile = null;
let uploadedFilePath = null;
let analysisComplete = false;

// ===== WebSocket Events =====
socket.on('connect', () => {
    console.log('Connected to server');
    statusDot.classList.add('connected');
    statusText.textContent = 'Connected';
    addSystemMessage('Connected to Deepfake Detection Agent System');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Disconnected';
});

socket.on('status', (data) => {
    console.log('Status:', data);
});

socket.on('agent_status', (data) => {
    console.log('Agent status:', data);
    showAgentStatus(data.message);
});

socket.on('detection_result', (data) => {
    console.log('Detection result:', data);
    displayResults(data);
});

socket.on('analysis_result', (data) => {
    console.log('Analysis result:', data);
    displayExplanation(data.explanation);
    hideAgentStatus();
    analysisComplete = true;
    chatInput.disabled = false;
    sendBtn.disabled = false;
});

socket.on('question_answer', (data) => {
    console.log('Question answer:', data);
    addAgentMessage(data.answer);
    hideAgentStatus();
});

socket.on('error', (data) => {
    console.error('Error:', data);
    addSystemMessage(`Error: ${data.message}`, 'error');
    hideAgentStatus();
});

socket.on('session_saved', (data) => {
    console.log('Session saved:', data);
    addSystemMessage(`Session saved: ${data.session_id}`);
});

// ===== File Upload =====
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearUpload();
});

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, WEBP, GIF)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        document.querySelector('.upload-content').style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    uploadedFile = null;
    uploadedFilePath = null;
    fileInput.value = '';
    document.querySelector('.upload-content').style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsCard.style.display = 'none';
    analysisComplete = false;
    chatInput.disabled = true;
    sendBtn.disabled = true;
}

// ===== Analyze Image =====
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';

    // Upload file
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadedFilePath = data.filepath;
            analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyzing...';

            // Clear previous results
            resultsCard.style.display = 'none';

            // Add user message
            addUserMessage(`Analyzing image: ${uploadedFile.name}`);

            // Send to agent via WebSocket
            socket.emit('analyze_image', {
                filepath: uploadedFilePath,
                query: ''
            });
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        addSystemMessage(`Upload failed: ${error.message}`, 'error');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Image';
    }
});

// ===== Chat =====
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !chatInput.disabled) {
        sendMessage();
    }
});

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Send question to agent
    socket.emit('ask_question', { question: message });
}

// ===== Message Display =====
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addAgentMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    chatInput.disabled = false;
    sendBtn.disabled = false;
}

function addSystemMessage(text, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ===== Results Display =====
function displayResults(data) {
    resultsCard.style.display = 'block';

    // Prediction label
    const predictionLabel = document.getElementById('predictionLabel');
    predictionLabel.textContent = data.label;
    predictionLabel.className = 'prediction-label ' + data.label.toLowerCase();

    // Confidence
    const confidence = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceValue').textContent = confidence + '%';

    // Confidence meter
    const meterFill = document.getElementById('meterFill');
    setTimeout(() => {
        meterFill.style.width = confidence + '%';
    }, 100);

    // Detailed scores
    const detailedScores = document.getElementById('detailedScores');
    detailedScores.innerHTML = '';

    for (const [label, score] of Object.entries(data.all_scores)) {
        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        scoreItem.innerHTML = `
            <span class="score-label">${label}</span>
            <span class="score-value">${(score * 100).toFixed(2)}%</span>
        `;
        detailedScores.appendChild(scoreItem);
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Reset analyze button
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Another';
}

function displayExplanation(text) {
    const explanationSection = document.getElementById('explanationSection');
    const explanationText = document.getElementById('explanationText');

    explanationSection.style.display = 'block';
    explanationText.textContent = text;

    // Add to chat
    addAgentMessage(text);
}

// ===== Agent Status =====
function showAgentStatus(message) {
    agentStatusText.textContent = message;
    agentStatus.style.display = 'block';
}

function hideAgentStatus() {
    agentStatus.style.display = 'none';
}

// ===== Utility Functions =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Initialize =====
console.log('Deepfake Detection Web App initialized');
