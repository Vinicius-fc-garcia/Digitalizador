import cv2
import numpy as np

def process_image(image):
    """
    Digitalizador robusto para documentos com fundos escuros, sombras e bordas complexas.
    Detecta documentos brancos mesmo em superfícies pretas ou com iluminação irregular.
    """
    orig = image.copy()
    h, w = image.shape[:2]
    
    # Redimensiona mantendo proporção para processamento mais rápido
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ratio = 1 / scale
    else:
        ratio = 1.0
    
    # Conversão para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica CLAHE para melhorar contraste em áreas escuras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # Blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    
    # Detecção de áreas CLARAS (papel branco) - método mais robusto
    # Usa threshold adaptativo para lidar com iluminação irregular
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Também usa Otsu para áreas uniformes
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combina os dois métodos
    thresh_combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
    
    # Operações morfológicas para limpar ruído e unir regiões
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    morph = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Detecta bordas com Canny
    edges = cv2.Canny(blurred, 30, 100)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Combina threshold e bordas
    combined = cv2.bitwise_or(morph, edges_dilated)
    
    # Encontra contornos
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Nenhum contorno encontrado")
        return orig
    
    # Filtra contornos por área (pelo menos 20% da imagem)
    img_area = image.shape[0] * image.shape[1]
    valid_contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.2]
    
    if not valid_contours:
        print("Nenhum contorno válido (área suficiente)")
        return orig
    
    # Pega o maior contorno
    cnt = max(valid_contours, key=cv2.contourArea)
    
    # Aproxima o contorno para encontrar o retângulo
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    # Tenta diferentes níveis de aproximação se não encontrar 4 pontos
    if len(approx) != 4:
        for epsilon_factor in [0.01, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
            if len(approx) == 4:
                break
    
    # Se ainda não tiver 4 pontos, usa bounding rect
    if len(approx) == 4:
        doc_cnt = approx.reshape(4, 2)
    else:
        print(f"Aproximação com {len(approx)} pontos, usando bounding rect")
        x, y, w, h = cv2.boundingRect(cnt)
        # Adiciona margem pequena
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        doc_cnt = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
    
    # Ordena pontos e aplica correção de perspectiva
    rect = order_points(doc_cnt.astype("float32"))
    rect *= ratio  # Ajusta para escala original
    
    (tl, tr, br, bl) = rect
    
    # Calcula dimensões do documento corrigido
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Pontos de destino
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Aplica transformação de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # Pós-processamento para melhorar qualidade
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Remove sombras com morphological transformation
    kernel_shadow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(warped_gray, cv2.MORPH_CLOSE, kernel_shadow)
    warped_no_shadow = cv2.divide(warped_gray, background, scale=255)
    
    # Aumenta contraste
    warped_enhanced = cv2.normalize(warped_no_shadow, None, 0, 255, cv2.NORM_MINMAX)
    
    # Aplica sharpening suave
    warped_sharp = cv2.addWeighted(warped_enhanced, 1.5, 
                                    cv2.GaussianBlur(warped_enhanced, (0, 0), 3), -0.5, 0)
    
    # Converte de volta para BGR
    return cv2.cvtColor(warped_sharp, cv2.COLOR_GRAY2BGR)


def order_points(pts):
    """
    Ordena pontos na ordem: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Soma: top-left terá menor soma, bottom-right terá maior soma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diferença: top-right terá menor diff, bottom-left terá maior diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


# Função auxiliar para teste
def test_scanner(image_path):
    """
    Testa o scanner e mostra resultado
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return
    
    result = process_image(image)
    
    # Mostra resultado
    cv2.imshow("Original", cv2.resize(image, (600, 800)))
    cv2.imshow("Digitalizado", cv2.resize(result, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result


# Exemplo de uso:
# img = cv2.imread('seu_documento.jpg')
# resultado = process_image(img)
# cv2.imwrite('documento_digitalizado.jpg', resultado)
