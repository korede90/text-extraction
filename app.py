from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pickle
import pytesseract
import re
import cv2
import json
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the pipeline
try:
    with open('ocr_pipeline.pkl', 'rb') as f:
        tesseract_cmd, amount_pattern = pickle.load(f)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
except FileNotFoundError:
    logging.error("OCR pipeline file (ocr_pipeline.pkl) not found!")
    exit(1)
except Exception as e:
    logging.error(f"Error loading OCR pipeline: {e}")
    exit(1)

def extract_amount(image_path):
    """Extract amounts from the uploaded image."""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return ['Error: Image not loaded']

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to enhance OCR accuracy
        scale_factor = 2  # Adjust as needed
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Reduce noise (optional, adjust kernel size as needed)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Extract text using PyTesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config)

        # Debug: Log raw OCR output
        logging.debug(f"Raw OCR Output: {text}")

        # Find amounts using regex
        # Updated regex pattern to capture amounts with commas (e.g., 3,600, 17,500)
        amount_pattern = re.compile(r'\d{1,3}(?:,\d{3})+')
        amounts = amount_pattern.findall(text)

        # Remove duplicates
        amounts = list(set(amounts))

        # # Sort amounts numerically
        # def get_numeric_value(amount):
        #     return int(amount.replace(',', ''))
        # amounts.sort(key=get_numeric_value)
        # Remove commas from the amounts
        amounts = [amount.replace(',', '') for amount in amounts]

        # Sort amounts numerically
        amounts.sort(key=lambda x: int(x))
        
        # Save extracted amounts to a file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_amounts.csv')
        with open(save_path, 'w') as f:
            for amount in amounts:
                f.write(f"{amount}\n")

        return amounts if amounts else ['No amount found']
    except Exception as e:
        logging.error(f"Error extracting amounts: {e}")
        return ['Error processing image']

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Upload file and extract amounts."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})

        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract amounts
        amounts = extract_amount(filepath)

        # Handle errors in extraction
        if 'Error' in amounts[0]:
            return jsonify({'error': amounts[0]})

        # Redirect to result page with JSON encoded amounts
        return redirect(url_for('result', image_path=file.filename, amounts=json.dumps(amounts)))

@app.route('/result')
def result():
    """Display extracted amounts."""
    image_path = request.args.get('image_path', None)
    amounts_json = request.args.get('amounts', '[]')
    extracted_amounts = json.loads(amounts_json)  # Correctly parse the JSON list

    return render_template('result.html', image_path='/' + os.path.join(app.config['UPLOAD_FOLDER'], image_path),
                           extracted_amounts=extracted_amounts)

if __name__ == '__main__':
    app.run(debug=True)