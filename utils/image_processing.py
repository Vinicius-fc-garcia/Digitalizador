import cv2
import numpy as np

def process_image(image):
    """
    Detecta o documento em uma foto, corrige perspectiva e recorta de forma segura.
    Mantém todo o conteúdo da folha visível, mesmo com fundo escuro ou colorido.
    """

    orig = image.copy()

    # Converter para tons de cinza e aumentar contraste
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)

    # Detectar bordas com Canny adaptativo
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edged = cv2.Canny(gray, lower, upper)

    # Dilatar e erodir para unir bordas
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    # Encontrar contornos externos
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return orig

    # Pegar o maior contorno com área mínima
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50000:  # evita recorte muito pequeno
        return orig

    # Aproximar contorno a um polígono de 4 lados (caso seja uma folha)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        doc_cnt = approx
    else:
        # fallback: usar bounding box (para garantir documento inteiro)
        x, y, w, h = cv2.boundingRect(cnt)
        doc_cnt = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    # Ordenar os pontos
    pts = doc_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calcular tamanho de saída
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Aplicar transformação de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Ajuste de contraste e remoção leve de sombras
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_norm = cv2.normalize(warped_gray, None, 0, 255, cv2.NORM_MINMAX)
    warped_final = cv2.convertScaleAbs(warped_norm, alpha=1.1, beta=15)

    return cv2.cvtColor(warped_final, cv2.COLOR_GRAY2BGR)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # topo esquerdo
    rect[2] = pts[np.argmax(s)]  # base direita

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # topo direito
    rect[3] = pts[np.argmax(diff)]  # base esquerda

    return rect
