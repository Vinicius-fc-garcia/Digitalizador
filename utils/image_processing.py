import cv2
import numpy as np

def process_image(image):
    """
    Digitalizador aprimorado — detecta documento mesmo sob sombras ou baixa iluminação.
    Usa detecção híbrida de bordas + área clara e corrige perspectiva automaticamente.
    """

    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta áreas claras (papel)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_white = cv2.inRange(thresh, 200, 255)

    # Combina bordas + branco
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(edges, mask_white)

    # Encontrar contornos
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return orig

    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        doc_cnt = approx
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        doc_cnt = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    # Ordena pontos e aplica correção
    pts = doc_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect * ratio

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Ajustes finais (contraste e nitidez)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    warped_norm = cv2.normalize(warped_gray, None, 0, 255, cv2.NORM_MINMAX)
    warped_final = cv2.convertScaleAbs(warped_norm, alpha=1.3, beta=10)

    return cv2.cvtColor(warped_final, cv2.COLOR_GRAY2BGR)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
