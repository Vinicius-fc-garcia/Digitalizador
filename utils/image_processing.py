import cv2
import numpy as np

def process_image(image):
    """
    Digitalizador que detecta, corrige perspectiva e enquadra documentos.
    Remove completamente as bordas e corrige distorções.
    """
    orig = image.copy()
    h, w = image.shape[:2]
    
    # Redimensiona para processamento
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ratio = 1 / scale
    else:
        image_resized = image.copy()
        ratio = 1.0
    
    # Encontra o contorno do documento
    doc_contour = find_document_contour(image_resized)
    
    if doc_contour is None:
        print("⚠️ Não foi possível detectar o documento")
        return orig
    
    # Ajusta para escala original
    doc_contour = doc_contour * ratio
    
    # Ordena os pontos
    pts = order_points(doc_contour)
    
    # Calcula as dimensões do documento retificado
    (tl, tr, br, bl) = pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destino retangular
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Aplica transformação de perspectiva
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # Pós-processamento para melhorar qualidade
    warped = enhance_document(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px")
    return warped


def find_document_contour(image):
    """
    Detecta o contorno do documento usando múltiplas técnicas.
    Retorna os 4 cantos do documento.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Melhora contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # --- Método 1: Threshold Adaptativo ---
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # --- Método 2: Otsu ---
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- Método 3: Canny ---
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Combina os métodos
    combined = cv2.bitwise_or(thresh1, thresh2)
    combined = cv2.bitwise_or(combined, edges)
    
    # Operações morfológicas para conectar bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilata para garantir contorno fechado
    combined = cv2.dilate(combined, kernel, iterations=2)
    
    # Encontra contornos
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Ordena por área
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = image.shape[0] * image.shape[1]
    
    # Procura o melhor contorno retangular
    for cnt in contours[:5]:  # Testa os 5 maiores
        area = cv2.contourArea(cnt)
        
        # Deve ter pelo menos 15% da área da imagem
        if area < img_area * 0.15:
            continue
        
        # Aproxima o contorno
        peri = cv2.arcLength(cnt, True)
        
        # Tenta diferentes níveis de aproximação
        for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            
            if len(approx) == 4:
                # Verifica se é um quadrilátero convexo
                if cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype("float32")
                
                # Se não for convexo, ainda pode servir
                return approx.reshape(4, 2).astype("float32")
    
    # Se não encontrou 4 pontos, usa o maior contorno e cria retângulo
    if contours:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return box.astype("float32")
    
    return None


def order_points(pts):
    """
    Ordena pontos: top-left, top-right, bottom-right, bottom-left
    """
    # Inicializa array de pontos ordenados
    rect = np.zeros((4, 2), dtype="float32")
    
    # Soma de coordenadas
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    
    # Diferença de coordenadas
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    
    return rect


def enhance_document(image):
    """
    Melhora a qualidade do documento digitalizado.
    Remove sombras, aumenta contraste e nitidez.
    """
    # Converte para escala de cinza
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Remove sombras usando morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Divide a imagem pelo fundo para normalizar iluminação
    normalized = cv2.divide(gray, background, scale=255)
    
    # Aumenta contraste
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold adaptativo para binarizar (opcional, comentado por padrão)
    # Se quiser documento completamente preto e branco, descomente:
    # normalized = cv2.adaptiveThreshold(
    #     normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY, 11, 10
    # )
    
    # Aplica unsharp mask para aumentar nitidez
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 2.0)
    sharpened = cv2.addWeighted(normalized, 1.5, gaussian, -0.5, 0)
    
    # Ajusta brilho e contraste final
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=5)
    
    # Converte de volta para BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def draw_contour_debug(image, contour):
    """
    Função auxiliar para debug - desenha o contorno detectado.
    """
    debug_img = image.copy()
    if contour is not None:
        pts = contour.astype(np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
        for i, pt in enumerate(pts):
            cv2.circle(debug_img, tuple(pt), 10, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), tuple(pt), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return debug_img


# Exemplo de uso completo
def scan_document(input_path, output_path=None, show_debug=False):
    """
    Função completa para escanear um documento.
    
    Args:
        input_path: Caminho da imagem de entrada
        output_path: Caminho para salvar (opcional)
        show_debug: Se True, mostra imagem com contorno detectado
    
    Returns:
        Imagem digitalizada
    """
    # Carrega imagem
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Erro ao carregar: {input_path}")
        return None
    
    print(f"📄 Processando: {input_path}")
    print(f"   Dimensões originais: {image.shape[1]}x{image.shape[0]}px")
    
    # Processa
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"💾 Salvo em: {output_path}")
    
    # Mostra debug se solicitado
    if show_debug:
        cv2.imshow("Original", cv2.resize(image, (800, 600)))
        cv2.imshow("Digitalizado", cv2.resize(result, (800, 600)))
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# Uso direto:
# resultado = process_image(cv2.imread('documento.jpg'))
# cv2.imwrite('documento_digitalizado.jpg', resultado)

# Ou usar a função completa:
# scan_document('documento.jpg', 'saida.jpg', show_debug=True)
