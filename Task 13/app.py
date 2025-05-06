from flask import Flask, request, render_template, send_file
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  
ENHANCED_FOLDER = 'enhanced'  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
os.makedirs(ENHANCED_FOLDER, exist_ok=True)  

def enhance_image(input_path, output_path):
    image = cv2.imread(input_path)  
    if image is None:
        raise ValueError("Failed to read image file.")  
    enhanced = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.20)  
    cv2.imwrite(output_path, enhanced)  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  
        file = request.files['image']  
        if file:
            filename = file.filename  
            input_path = os.path.join(UPLOAD_FOLDER, filename) 
            output_path = os.path.join(ENHANCED_FOLDER, f"enhanced_{filename}")  
            file.save(input_path)  
            enhance_image(input_path, output_path) 
            return send_file(output_path, mimetype='image/jpeg')  
    return render_template('index.html')  

app.run(debug=True)  
