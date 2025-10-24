import cv2
import numpy as np

def process_image(image):
    """
    Detecta automaticamente o contorno da folha de papel, corrige perspectiva,
    corta as bordas e melhora a nitidez/contraste.
    """

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aumentar contraste e reduzir ruído
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Binarizar (melhor para bordas de documentos)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar bordas
    edged = cv2.Canny(thresh, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    # Encontrar contornos
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image

    # Pegar o maior contorno (provável folha)
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        doc_cnt = approx
    else:
        # fallback: retângulo delimitador
        x, y, w, h = cv2.boundingRect(cnt)
        doc_cnt = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    # Aplicar transformação de perspectiva
    pts = doc_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

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

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Melhorar contraste e nitidez do resultado
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.convertScaleAbs(warped_gray, alpha=1.3, beta=15)
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # topo esquerdo
    rect[2] = pts[np.argmax(s)]  # base direita

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # topo direito
    rect[3] = pts[np.argmax(diff)]  # base esquerda

    return rect
