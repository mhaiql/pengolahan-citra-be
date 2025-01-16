from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Konstanta untuk tipe MIME
MIME_PNG = 'image/png'
MIME_JPEG = 'image/jpeg'


def read_image(file):
    """Fungsi untuk membaca gambar dari file."""
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

def convert_image_to_buffer(image, format='jpg'):
    """Fungsi untuk mengonversi gambar ke buffer."""
    if format == 'png':
        _, buffer = cv2.imencode('.png', image)
    else:  # default to jpg
        _, buffer = cv2.imencode('.jpg', image)
    return io.BytesIO(buffer)

@app.route('/')
def home():
    """Route default yang mengembalikan pesan layanan."""
    return "SERVICE BACKEND PYTHON V.1"

@app.route('/grayscale', methods=['POST'])
def convert_to_grayscale():
    try:
        file = request.files['image']
        img = read_image(file)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Dapatkan ekstensi file untuk menentukan format output
        ext = os.path.splitext(file.filename)[1].lower()
        format = 'png' if ext == '.png' else 'jpg'  # Default ke jpg jika bukan png
        
        # Konversi gambar ke buffer sesuai format
        io_buf = convert_image_to_buffer(gray_img, format=format)
        
        return send_file(io_buf, mimetype=MIME_PNG if format == 'png' else MIME_JPEG)
    except ValueError as e:
        return {"error": str(e)}, 400
    
@app.route('/blur_edges', methods=['POST'])  
def blur_edges():  
    try:  
        file = request.files['image']  
        img = read_image(file)  
          
        # Terapkan Gaussian blur ke seluruh gambar  
        blurred_img = cv2.GaussianBlur(img, (61, 61), 0)  
          
        # Dapatkan ekstensi file untuk menentukan format output  
        ext = os.path.splitext(file.filename)[1].lower()  
        format = 'png' if ext == '.png' else 'jpg'  # Default ke jpg jika bukan png  
          
        # Konversi gambar ke buffer sesuai format  
        io_buf = convert_image_to_buffer(blurred_img, format=format)  
          
        return send_file(io_buf, mimetype='image/png' if format == 'png' else 'image/jpeg', as_attachment=True, download_name='output_blur_edges' + ext)  
    except ValueError as e:  
        return {"error": str(e)}, 400  


# @app.route('/blur_edges', methods=['POST'])
# def blur_edges():
#     try:
#         file = request.files['image']
#         img = read_image(file)
        
#         h, w = img.shape[:2]
        
#         edge_thickness_h = int(h * 0.15)
#         edge_thickness_w = int(w * 0.25) 
        
#         # Create a mask for the edges
#         mask = np.zeros((h, w), dtype=np.uint8)
        
#         # Define the area to keep sharp (the center of the image)
#         mask[edge_thickness_h:h-edge_thickness_h, edge_thickness_w:w-edge_thickness_w] = 255
        
#         # Apply Gaussian blur to the entire image
#         blurred_img = cv2.GaussianBlur(img, (61, 61), 0)
        
#         # Combine the original image and the blurred image using the mask
#         result = np.where(mask[:, :, np.newaxis] == 255, img, blurred_img)
        
#         # Get the file extension to determine output format
#         ext = os.path.splitext(file.filename)[1].lower()
#         format = 'png' if ext == '.png' else 'jpg'  # Default to jpg if not png
        
#         # Convert the image to buffer according to format
#         io_buf = convert_image_to_buffer(result, format=format)
        
#         return send_file(io_buf, mimetype='image/png' if format == 'png' else 'image/jpeg', as_attachment=True, download_name='output_blur_edges' + ext)
#     except ValueError as e:
#         return {"error": str(e)}, 400

@app.route('/resize', methods=['POST'])
def resize_image():
    try:
        file = request.files['image']
        img = read_image(file)
        
        h, w = img.shape[:2]
        percentage = request.form.get('percentage', default=50, type=int)
        
        new_width = int(w * (percentage / 100))
        new_height = int(h * (percentage / 100))
        new_size = (new_width, new_height)
        
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Dapatkan ekstensi file untuk menentukan format output
        ext = os.path.splitext(file.filename)[1].lower()
        format = 'png' if ext == '.png' else 'jpg'  # Default ke jpg jika bukan png
        
        # Konversi gambar ke buffer sesuai format
        io_buf = convert_image_to_buffer(resized_img, format=format)
        
        return send_file(io_buf, mimetype=MIME_PNG if format == 'png' else MIME_JPEG, as_attachment=True, download_name='output_resized' + ext)
    except ValueError as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True)
