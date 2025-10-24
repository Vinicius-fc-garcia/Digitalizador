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
        ratio = scale
    else:
        image_resized = image.copy()
        ratio = 1.0
    
    # Encontra o contorno do documento
    doc_contour = find_document_contour(image_resized)
    
    if doc_contour is None:
        # Tenta usar a imagem inteira como fallback se nada for detectado
        print("⚠️ Não foi possível detectar o documento, usando imagem inteira.")
        pts = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")
        
        # Calcula dimensões
        maxWidth = w
        maxHeight = h
        
        # Define destino
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

    else:
        # Se detectou, ajusta para escala original
        doc_contour = doc_contour / ratio
        
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
    # A linha 'pts = order_points(pts)' foi REMOVIDA daqui pois era redundante.
    # 'pts' já foi ordenado logo após a detecção do contorno.
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
        cv2.THRESH_BINARY_INV, 21, 5 # Invertido para bordas brancas
    )
    
    # --- Método 2: Canny ---
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Combina os métodos
    combined = cv2.bitwise_or(thresh1, edges)
    
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
    for cnt in contours[:5]: # Testa os 5 maiores
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
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    
    # Diferença de coordenadas (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def enhance_document(image):
    """
    ✅ CORRIGIDO: Melhora a qualidade do documento digitalizado.
    Remove sombras, aumenta contraste e nitidez.
    Agora preserva a cor se o original for colorido.
    """
    
    # Se for colorido, processa o canal de Luminosidade (L)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Converte para LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Remove sombras do canal L
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        l_normalized = cv2.divide(l, background, scale=255)
        
        # Aumenta contraste do canal L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_normalized)
        
        # Mescla canais de volta
        merged = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Aplica nitidez na imagem colorida
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Ajuste final de brilho/contraste
        final = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=2)
        
        # --- OPÇÃO PARA IMPRESSÃO (PRETO E BRANCO) ---
        # Descomente as 3 linhas abaixo para P&B de alto contraste
        # print("Convertendo para P&B de alto contraste para impressão.")
        # gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        # _, final_bw = cv2.threshold(gray_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # final = cv2.cvtColor(final_bw, cv2.COLOR_GRAY2BGR) # Converte de volta para 3 canais
            
        return final
        
    else:
        # Se já for P&B, usa a lógica antiga
        gray = image.copy()
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.divide(gray, background, scale=255)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        # --- OPÇÃO PARA IMPRESSÃO (PRETO E BRANCO) ---
        # Descomente para P&B de alto contraste
        # print("Convertendo para P&B de alto contraste para impressão.")
        # normalized = cv2.adaptiveThreshold(
        #     normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #     cv2.THRESH_BINARY, 11, 10
        # )
        
        gaussian = cv2.GaussianBlur(normalized, (0, 0), 2.0)
        sharpened = cv2.addWeighted(normalized, 1.5, gaussian, -0.5, 0)
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=5)
        
        # Converte de volta para BGR (para consistência)
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
    print(f"    Dimensões originais: {image.shape[1]}x{image.shape[0]}px")
    
    # Processa
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Salvo em: {output_path}")
    
    # Mostra debug se solicitado
    if show_debug:
        h, w = image.shape[:2]
        ratio = 800 / max(h, w)
        preview_orig = cv2.resize(image, None, fx=ratio, fy=ratio)
        
        h_res, w_res = result.shape[:2]
        ratio_res = 800 / max(h_res, w_res)
        preview_res = cv2.resize(result, None, fx=ratio_res, fy=ratio_res)

        cv2.imshow("Original", preview_orig)
        cv2.imshow("Digitalizado", preview_res)
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# --- COMO USAR ---

# 1. Defina os caminhos
# Lembre-se de mudar 'documento.jpg' para o nome do seu arquivo
arquivo_entrada = 'documento.jpg'
arquivo_saida = 'documento_digitalizado.jpg'

# 2. Execute a digitalização
# scan_document(arquivo_entrada, arquivo_saida, show_debug=True)

# Exemplo simples (se você já tiver a imagem carregada):
# img = cv2.imread(arquivo_entrada)
# if img is not None:
#     resultado = process_image(img)
#     cv2.imwrite(arquivo_saida, resultado)
#     cv2.imshow("Resultado", cv2.resize(resultado, (800, 1000)))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print(f"Não foi possível carregar a imagem: {arquivo_entrada}")
