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
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/subir', methods=['POST'])
def subir_imagen():
    archivo = request.files['imagen']
    formato_salida = request.form.get('formato', 'jpg').lower()
    efecto_manual = request.form.get('efectoManual') or 'ninguno'
    brillo = int(request.form.get('brillo', 0))
    desenfoque = int(request.form.get('desenfoque', 15))
    marca_agua = request.form.get('marcaAgua', '')
    comprimir = request.form.get('comprimir') == 'true'
    ocr_activado = request.form.get('ocr') == 'true'
    desenfoque_rostros = request.form.get('desenfoqueRostros') == 'true'
    dibujar_contornos = request.form.get('contornos') == 'true'
    tipo_desenfoque_rostro = request.form.get('tipoDesenfoqueRostro', 'gaussian')

    datos_imagen = archivo.read()
    img_np = cv2.imdecode(np.frombuffer(datos_imagen, np.uint8), cv2.IMREAD_COLOR)
    original_np = img_np.copy()

    img_np = aplicar_efecto(img_np, efecto_manual, brillo, desenfoque)

    if dibujar_contornos:
        bordes = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY), 50, 150)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contornos, -1, (0, 255, 0), 2)

    if desenfoque_rostros:
        clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gris = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        rostros = clasificador_rostros.detectMultiScale(gris, 1.1, 4)
        for (x, y, w, h) in rostros:
            rostro = img_np[y:y+h, x:x+w]
            if tipo_desenfoque_rostro == 'gaussian':
                desenfocado = cv2.GaussianBlur(rostro, (61, 61), 40)
            elif tipo_desenfoque_rostro == 'pixelar':
                escala = 0.05
                pequeño = cv2.resize(rostro, (0, 0), fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
                desenfocado = cv2.resize(pequeño, (w, h), interpolation=cv2.INTER_NEAREST)
            elif tipo_desenfoque_rostro == 'media':
                desenfocado = cv2.blur(rostro, (40, 40))
            elif tipo_desenfoque_rostro == 'bilateral':
                desenfocado = cv2.bilateralFilter(rostro, 30, 90, 90)
            else:
                desenfocado = rostro
            img_np[y:y+h, x:x+w] = desenfocado

    if marca_agua:
        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        dibujar = ImageDraw.Draw(img_pil)
        fuente = ImageFont.load_default()
        dibujar.text((10, 10), marca_agua, font=fuente, fill=(255, 255, 255))
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if ocr_activado:
        gris = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        texto = pytesseract.image_to_string(gris, config='--psm 6')
        return texto, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    if formato_salida == 'svg':
        return convertir_a_svg_potrace(original_np)

    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    formato_salida_pil = 'JPEG' if formato_salida == 'jpg' else formato_salida.upper()
    img_pil.save(buffer, format=formato_salida_pil)
    buffer.seek(0)
    return send_file(buffer, mimetype=f'image/{formato_salida}')


def convertir_a_svg_potrace(img_np):
    id_temporal = str(uuid.uuid4())
    archivo_pbm = f"{id_temporal}.pbm"
    archivo_svg = f"{id_temporal}.svg"
    
    try:
        gris = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, bn = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY)
        Image.fromarray(bn).save(archivo_pbm)
        subprocess.run(['potrace', archivo_pbm, '-s', '-o', archivo_svg], check=True)
        with open(archivo_svg, "rb") as f:
            return send_file(BytesIO(f.read()), mimetype='image/svg+xml')
    finally:
        for archivo in [archivo_pbm, archivo_svg]:
            if os.path.exists(archivo):
                os.remove(archivo)


def aplicar_efecto(img, efecto, brillo=0, desenfoque=15):
    if brillo != 0:
        img = cv2.convertScaleAbs(img, alpha=1, beta=brillo)
    if desenfoque > 1:
        tamaño_kernel = max(1, int(desenfoque) // 2 * 2 + 1)
        img = cv2.GaussianBlur(img, (tamaño_kernel, tamaño_kernel), 0)
    if efecto == 'escala_grises':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif efecto == 'invertir':
        img = cv2.bitwise_not(img)
    elif efecto == 'sepia':
        matriz_sepia = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        img = cv2.transform(img, matriz_sepia)
    elif efecto == 'caricatura':
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gris_desenfocado = cv2.medianBlur(gris, 7)
        bordes = cv2.adaptiveThreshold(gris_desenfocado, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(img, 9, 250, 250)
        img = cv2.bitwise_and(color, color, mask=bordes)
    elif efecto == 'canny':
        img = cv2.Canny(img, 100, 200)
    return img

if __name__ == '__main__':
    app.run(debug=True)
