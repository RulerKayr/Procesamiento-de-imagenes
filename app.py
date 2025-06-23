from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pytesseract
import subprocess
import uuid
import os

app = Flask(__name__)

# Ruta a Tesseract en Windows (ajusta si es necesario)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    output_format = request.form.get('format', 'jpg').lower()
    manual_effect = request.form.get('manualEffect') or 'none'
    brightness = int(request.form.get('brightness', 0))
    blur = int(request.form.get('blur', 15))
    watermark = request.form.get('watermark', '')
    compress = request.form.get('compress') == 'true'
    ocr_enabled = request.form.get('ocr') == 'true'
    face_blur = request.form.get('faceBlur') == 'true'
    draw_contours = request.form.get('contours') == 'true'

    img_data = file.read()
    img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    original_np = img_np.copy()

    # Efectos en cascada
    img_np = apply_effect(img_np, manual_effect, brightness, blur)

    if draw_contours:
        edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contours, -1, (0, 255, 0), 2)

    if face_blur:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_region = img_np[y:y+h, x:x+w]
            face_blurred = cv2.GaussianBlur(face_region, (61, 61), 40)
            img_np[y:y+h, x:x+w] = face_blurred

    if watermark:
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        draw.text((10, 10), watermark, font=font, fill=(255, 255, 255))
        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if ocr_enabled:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    if output_format == 'svg':
        return convert_to_svg_potrace(original_np)

    pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    format_out = 'JPEG' if output_format == 'jpg' else output_format.upper()
    pil_img.save(buf, format=format_out)
    buf.seek(0)
    return send_file(buf, mimetype=f'image/{output_format}')

def convert_to_svg_potrace(img_np):
    temp_id = str(uuid.uuid4())
    temp_pbm = f"{temp_id}.pbm"
    temp_svg = f"{temp_id}.svg"
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        Image.fromarray(bw).save(temp_pbm)
        subprocess.run(['potrace', temp_pbm, '-s', '-o', temp_svg], check=True)
        with open(temp_svg, "rb") as f:
            return send_file(BytesIO(f.read()), mimetype='image/svg+xml')
    finally:
        for f in [temp_pbm, temp_svg]:
            if os.path.exists(f):
                os.remove(f)

def apply_effect(img, effect, brightness=0, blur=15):
    # Aplicar en cascada según los efectos activados
    if brightness != 0:
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    if blur > 1:
        ksize = max(1, int(blur) // 2 * 2 + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if effect == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif effect == 'invert':
        img = cv2.bitwise_not(img)
    elif effect == 'sepia':
        sepia = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        img = cv2.transform(img, sepia)
    elif effect == 'cartoon':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur_gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(img, 9, 250, 250)
        img = cv2.bitwise_and(color, color, mask=edges)
    elif effect == 'canny':
        img = cv2.Canny(img, 100, 200)
    return img

if __name__ == '__main__':
    app.run(debug=True)
