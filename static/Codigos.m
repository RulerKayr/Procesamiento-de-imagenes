%% 1. Cargar imagen
img = imread('jojo.jpg');
img = im2double(img); % Escalar entre 0-1 para operaciones posteriores
imshow(img);
title('Imagen original');

%% 2. Escala de grises (Grayscale)
gray = rgb2gray(img);
imshow(gray);
title('Escala de grises');

%% 3. Invertir colores (Invertido)
inverted = imcomplement(img);
imshow(inverted);
title('Invertido');

%% 4. Brillo (ajuste lineal)
brillo = img + 0.5; % Aumenta brillo (ajusta valor según necesidad)
brillo = min(brillo, 1); % Limita a 1
imshow(brillo);
title('Brillo +0.2');

%% 5. Desenfoque (Gaussian blur)
h = fspecial('gaussian', [15 15], 1); % Tamaño y sigma
blurred = imfilter(img, h, 'replicate');
imshow(blurred);
title('Desenfoque Gaussiano');

%% 6. Detección de bordes (Canny)
gray = rgb2gray(img);
edges = edge(gray, 'Canny');
imshow(edges);
title('Bordes - Canny');

%% 7. Filtro Sepia
sepia_filter = [0.393, 0.769, 0.189;
                0.349, 0.686, 0.168;
                0.272, 0.534, 0.131];
sepia = reshape(img, [], 3) * sepia_filter';
sepia = reshape(sepia, size(img));
sepia = min(sepia, 1);
imshow(sepia);
title('Sepia');

%% 8. Estilo Cartoon (modo básico con bordes)
gray = rgb2gray(img);
edges = edge(gray, 'Canny');
edges = imcomplement(edges);
cartoon = img .* repmat(edges, [1 1 3]);
imshow(cartoon);
title('Cartoon');

%% 9. Umbralización para SVG (binaria)
gray = rgb2gray(img);
bw = imbinarize(gray, 0.5); % 0.5 equivale a umbral 128 en 0-255
imshow(bw);
title('Umbralización binaria');

%% 10. Histograma (de imagen en escala de grises)
gray = rgb2gray(img);
imhist(gray);
title('Histograma de niveles de gris');
%% 11. Detección de rostros con desenfoque (tipo privacidad)
img=imread("cara_1_foto_70.jpg");
% Asegúrate de tener la Vision Toolbox
faceDetector = vision.CascadeObjectDetector();

% Detectar rostros
bbox = step(faceDetector, img);

% Copia de la imagen para aplicar desenfoque sobre los rostros
img_faces = img;

% Aplicar desenfoque a cada rostro detectado
for i = 1:size(bbox, 1)
    x = bbox(i, 1); y = bbox(i, 2);
    w = bbox(i, 3); h = bbox(i, 4);
    
    % Extraer región del rostro
    rostro = img_faces(y:y+h-1, x:x+w-1, :);
    
    % Aplicar desenfoque (ajusta tamaño y sigma según lo fuerte que quieras)
    h_blur = fspecial('gaussian', [61 61], 20);
    rostro_blur = imfilter(rostro, h_blur, 'replicate');
    
    % Reemplazar rostro desenfocado en imagen original
    img_faces(y:y+h-1, x:x+w-1, :) = rostro_blur;
end

imshow(img_faces);
title('Desenfoque de rostros (privacidad)');

%% 12. Reconocimiento de texto (OCR)

% Convertir a escala de grises
img=imread("letras.jpg");
gray = rgb2gray(img);

% Opcional: mejorar contraste con ecualización
gray_eq = adapthisteq(gray);

% Aplicar binarización adaptativa
bw = imbinarize(gray_eq, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.5);

% Ejecutar OCR
results = ocr(bw);

% Mostrar texto reconocido
disp('Texto extraído por OCR:');
disp(results.Text);

% Mostrar visualización sobre la imagen original
ocrOverlay = insertText(img, [10 10], results.Text, 'FontSize', 18, 'BoxColor', 'green', 'TextColor', 'white');
imshow(ocrOverlay);
title('Texto extraído (OCR)');