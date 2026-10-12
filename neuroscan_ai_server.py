import os
import io
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask application
app = Flask(__name__)

# Define the path to your trained SVM model
MODEL_PATH = 'svm_model.pkl'

# MUST MATCH YOUR PYTHON TRAINING LABELS EXACTLY (Order matters)
CLASS_LABELS = ["Pituitary", "No Tumor", "Meningioma", "Glioma"]

# Try to load the model, fallback to a dummy model for demonstration if file is missing
try:
    with open(MODEL_PATH, 'rb') as f:
        svm_model = pickle.load(f)
    print("✅ SVM Model loaded successfully.")
    USING_MOCK = False
except FileNotFoundError:
    print(f"⚠️ Warning: '{MODEL_PATH}' not found.")
    print("⚠️ Running in MOCK mode. Please place your trained .pkl file in the same directory.")
    svm_model = None
    USING_MOCK = True

# We embed your provided HTML directly into the Python file to keep this a single-file application.
# The JavaScript has been updated to use the Fetch API to communicate with this Python server.
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI - Brain Tumor Predictor (SVM Backend)</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts for a tech/medical feel -->
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    
    <style>
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: #0f172a; 
            color: #e2e8f0;
            overflow-x: hidden;
        }
        h1, h2, h3, .brand-font {
            font-family: 'Rajdhani', sans-serif;
        }
        
        /* Glassmorphism Panel */
        .glass-panel {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        /* Custom File Drop Zone */
        .drop-zone {
            border: 2px dashed rgba(6, 182, 212, 0.4); 
            transition: all 0.3s ease;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #06b6d4;
            background: rgba(6, 182, 212, 0.05);
            box-shadow: inset 0 0 20px rgba(6, 182, 212, 0.1);
        }

        /* Image Wrapper & Scanner Animation */
        .image-wrapper {
            position: relative;
            overflow: hidden;
            border-radius: 0.5rem;
            display: none;
            border: 1px solid rgba(255,255,255,0.1);
        }
        #image-preview {
            display: block;
            width: 100%;
            height: auto;
            max-height: 350px;
            object-fit: contain;
            background: #000;
        }
        
        /* The laser scanning line */
        .scanner-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: #06b6d4;
            box-shadow: 0 0 10px #06b6d4, 0 0 20px #06b6d4;
            z-index: 10;
            display: none;
            animation: scan 2s linear infinite;
        }
        .scanner-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(6, 182, 212, 0.15);
            display: none;
            z-index: 9;
        }

        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        /* Background animated orbs */
        .bg-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: -1;
            opacity: 0.4;
        }
        .orb-1 {
            background: #3b82f6; 
            width: 400px; height: 400px;
            top: -100px; left: -100px;
            animation: float 10s ease-in-out infinite alternate;
        }
        .orb-2 {
            background: #06b6d4; 
            width: 300px; height: 300px;
            bottom: 0px; right: -50px;
            animation: float 8s ease-in-out infinite alternate-reverse;
        }

        @keyframes float {
            0% { transform: translate(0, 0); }
            100% { transform: translate(30px, 30px); }
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4 relative">

    <!-- Background Orbs -->
    <div class="bg-orb orb-1"></div>
    <div class="bg-orb orb-2"></div>

    <div class="glass-panel p-8 rounded-2xl max-w-lg w-full relative z-10">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="flex justify-center items-center gap-2 mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                <h1 class="text-3xl font-bold text-white tracking-wide">NeuroScan <span class="text-cyan-400">AI</span></h1>
            </div>
            <p class="text-slate-400 text-sm">SVM Powered MRI Inference Backend</p>
        </div>

        <!-- File Upload Area -->
        <div class="mb-6">
            <input type="file" id="image-upload" accept="image/*" class="hidden">
            <label for="image-upload" id="drop-zone-label" class="drop-zone flex flex-col items-center justify-center w-full h-32 rounded-xl cursor-pointer bg-slate-800/50">
                <div class="flex flex-col items-center justify-center pt-5 pb-6">
                    <svg class="w-8 h-8 mb-3 text-cyan-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                        <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                    </svg>
                    <p class="mb-2 text-sm text-slate-300"><span class="font-semibold text-cyan-400">Click to upload</span> or drag and drop</p>
                    <p class="text-xs text-slate-500">MRI Scan Image (JPG/PNG)</p>
                </div>
            </label>
        </div>

        <div class="mb-6 text-center">
            <div id="image-wrapper" class="image-wrapper mx-auto">
                <img id="image-preview" src="" alt="MRI Scan">
                <div id="scanner-overlay" class="scanner-overlay"></div>
                <div id="scanner-line" class="scanner-line"></div>
            </div>
        </div>

        <!-- Action Button -->
        <button id="predict-btn" class="brand-font w-full bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-3 px-4 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_15px_rgba(6,182,212,0.3)] hover:shadow-[0_0_25px_rgba(6,182,212,0.6)] text-lg uppercase tracking-wider" disabled>
            Initiate Scan Analysis
        </button>

        <div id="result-container" class="mt-8 hidden">
            <div class="bg-slate-800/80 rounded-xl p-5 border border-slate-700">
                <h3 class="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-700 pb-2">Analysis Results</h3>
                
                <div class="flex justify-between items-center mb-2">
                    <span class="text-slate-300">Detected Classification:</span>
                    <span id="diagnosis" class="brand-font text-xl font-bold text-white tracking-wide"></span>
                </div>
                
                <div class="flex justify-between items-center">
                    <span class="text-slate-300">AI Confidence Level:</span>
                    <span id="confidence" class="brand-font font-bold text-cyan-400 text-lg drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]"></span>
                </div>
            </div>
        </div>
        
        <div id="error-message" class="mt-4 bg-red-900/30 border border-red-500/50 text-red-200 p-3 rounded-lg text-sm text-center hidden"></div>
    </div>

    <script>
        // DOM Elements
        const imageUpload = document.getElementById('image-upload');
        const imagePreview = document.getElementById('image-preview');
        const imageWrapper = document.getElementById('image-wrapper');
        const dropZone = document.getElementById('drop-zone-label');
        const predictBtn = document.getElementById('predict-btn');
        const resultContainer = document.getElementById('result-container');
        const diagnosisEl = document.getElementById('diagnosis');
        const confidenceEl = document.getElementById('confidence');
        const errorEl = document.getElementById('error-message');
        const scannerLine = document.getElementById('scanner-line');
        const scannerOverlay = document.getElementById('scanner-overlay');

        let selectedFile = null;

        // Handle file selection
        imageUpload.addEventListener('change', handleFileSelect);

        // Drag and drop functionality
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                imageUpload.files = e.dataTransfer.files;
                handleFileSelect({ target: imageUpload });
            }
        });

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imageWrapper.style.display = 'block'; 
                    predictBtn.disabled = false;
                    predictBtn.innerText = "Initiate Scan Analysis";
                    
                    // Reset UI states
                    resultContainer.classList.add('hidden');
                    errorEl.classList.add('hidden');
                    stopScanner();
                };
                reader.readAsDataURL(file);
            }
        }

        function startScanner() {
            scannerLine.style.display = 'block';
            scannerOverlay.style.display = 'block';
            predictBtn.innerText = "Analyzing Server-Side...";
            predictBtn.disabled = true;
        }

        function stopScanner() {
            scannerLine.style.display = 'none';
            scannerOverlay.style.display = 'none';
            predictBtn.innerText = "Initiate Scan Analysis";
            predictBtn.disabled = false;
        }

        // Send data to the Python Backend API
        predictBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            startScanner();
            resultContainer.classList.add('hidden');
            errorEl.classList.add('hidden');

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                // Call the Flask API
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || "Server Error");
                }

                const data = await response.json();
                
                // Process and display results from Python Backend
                const predictedClass = data.prediction;
                
                diagnosisEl.innerText = predictedClass === "No Tumor" ? "No Tumor Detected" : predictedClass + " Tumor";
                
                // Styling based on result
                if (predictedClass === "No Tumor") {
                    diagnosisEl.className = "brand-font text-2xl font-bold text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]";
                } else {
                    diagnosisEl.className = "brand-font text-2xl font-bold text-rose-500 drop-shadow-[0_0_8px_rgba(244,63,94,0.8)]";
                }
                
                confidenceEl.innerText = data.confidence;
                resultContainer.classList.remove('hidden');

            } catch (error) {
                console.error("Inference error:", error);
                showError(error.message || "Critical error during inference protocol.");
            } finally {
                stopScanner();
            }
        });

        // Helper function for errors
        function showError(msg) {
            errorEl.innerText = msg;
            errorEl.classList.remove('hidden');
            stopScanner();
        }
    </script>
</body>
</html>
"""

def preprocess_image_for_svm(image_bytes):
    """
    CRITICAL STEP: 
    SVMs require exact 1D arrays matching the shape they were trained on.
    You MUST modify this function to match the exact preprocessing steps 
    you used in your Jupyter Notebook/Python training script.
    """
    try:
        # 1. Open the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # 2. Convert to Grayscale or RGB (Uncomment whichever you used for training)
        img = img.convert('L') # 'L' is Grayscale, 'RGB' is Color
        
        # 3. Resize to the EXACT dimensions you used when training your SVM
        # Replace (128, 128) with your training dimensions (e.g., 64x64, 256x256)
        img = img.resize((128, 128)) 
        
        # 4. Convert to Numpy Array
        img_array = np.array(img)
        
        # 5. Flatten the image for the SVM (SVMs expect 2D arrays like (n_samples, n_features))
        # This converts a 128x128 image into a 1D array of 16,384 features, then wraps it in a list.
        flat_array = img_array.flatten().reshape(1, -1)
        
        # 6. Normalize pixel values (ONLY if you divided by 255.0 during training)
        flat_array = flat_array / 255.0 
        
        return flat_array
    except Exception as e:
        print("Error during preprocessing:", e)
        return None

@app.route('/')
def home():
    # Serve the HTML frontend
    return render_template_string(INDEX_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Ensure a file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        # 2. Read the image bytes
        image_bytes = file.read()
        
        # 3. Preprocess the image
        features = preprocess_image_for_svm(image_bytes)
        
        if features is None:
             return jsonify({'error': 'Failed to process image.'}), 500

        # 4. Make Prediction
        if USING_MOCK:
            # Fake prediction for demonstration if no model is loaded
            import time
            import random
            time.sleep(1.5) # simulate processing time
            predicted_index = random.randint(0, 3)
            confidence_str = f"{random.uniform(85.0, 99.9):.2f}% (Mock)"
        else:
            # REAL SVM PREDICTION
            prediction = svm_model.predict(features)
            predicted_index = int(prediction[0])
            
            # Note: For confidence, your SVM must have been trained with `probability=True`
            # (e.g., SVC(probability=True)). If not, predict_proba will throw an error.
            try:
                probabilities = svm_model.predict_proba(features)[0]
                confidence = float(np.max(probabilities)) * 100
                confidence_str = f"{confidence:.2f}%"
            except AttributeError:
                # Fallback if probability=True wasn't set during training
                confidence_str = "N/A (Model lacks predict_proba)"

        # 5. Get the label string
        if predicted_index < len(CLASS_LABELS):
             result_label = CLASS_LABELS[predicted_index]
        else:
             result_label = "Unknown Class"

        # 6. Return JSON response to the frontend
        return jsonify({
            'prediction': result_label,
            'confidence': confidence_str
        })
        
    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting NeuroScan Server on port {port}...")
    # host='0.0.0.0' is required to expose the app to the external web in cloud environments
    app.run(host='0.0.0.0', port=port)