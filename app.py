from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import svgwrite

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/google07bf63060c93272a.html')
def google_verification():
    return render_template('google07bf63060c93272a.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    effect = request.form.get('effect')
    output_format = request.form.get('format').lower()
    brightness = int(request.form.get('brightness', 0))
    blur = int(request.form.get('blur', 15))

    img_np = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = apply_effect(img_np, effect, brightness, blur)

    if output_format == 'svg':
        svg_bytes = raster_to_svg_bw(result)
        return send_file(svg_bytes, mimetype='image/svg+xml')

    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    format_map = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'bmp': 'BMP',
        'gif': 'GIF',
        'tiff': 'TIFF',
        'webp': 'WEBP'
    }

    pil_format = format_map.get(output_format)
    if not pil_format:
        return f"Formato '{output_format}' no soportado", 400

    output_buffer = BytesIO()
    pil_img.save(output_buffer, format=pil_format)
    output_buffer.seek(0)
    return send_file(output_buffer, mimetype=f'image/{output_format}')


def raster_to_svg_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    dwg = svgwrite.Drawing(size=(width, height))

    for cnt in contours:
        if len(cnt) >= 3:
            path_data = "M " + " L ".join(f"{pt[0][0]},{pt[0][1]}" for pt in cnt) + " Z"
            dwg.add(dwg.path(d=path_data, fill='black', stroke='black', stroke_width=1))

    svg_string = dwg.tostring()
    svg_buffer = BytesIO()
    svg_buffer.write(svg_string.encode('utf-8'))
    svg_buffer.seek(0)
    return svg_buffer


def apply_effect(img, effect, brightness=0, blur=15):
    if effect == 'grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif effect == 'canny':
        return cv2.Canny(img, 100, 200)
    elif effect == 'contours':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        canvas = img.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        return canvas
    elif effect == 'blur':
        ksize = max(1, int(blur) // 2 * 2 + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif effect == 'brightness':
        return cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    elif effect == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif effect == 'invert':
        return cv2.bitwise_not(img)
    elif effect == 'sepia':
        sepia = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(img, sepia)
    elif effect == 'cartoon':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    elif effect == 'colormap':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif effect == 'face':
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img
    else:
        return img


if __name__ == '__main__':
    app.run(debug=True)
