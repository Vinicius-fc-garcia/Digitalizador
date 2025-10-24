import cv2
import numpy as np

def order_points(pts):
    """Ordena pontos: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_document_contour(image):
    """
    CORREÇÃO V2: Detecta o documento COMPLETO incluindo a parte manuscrita.
    Usa detecção de área clara (papel branco) em vez de apenas bordas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. MÉTODO PRINCIPAL: Detectar áreas CLARAS (papel branco)
    # Isso pega todo o documento, incluindo onde tem caligrafia
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Detectar regiões claras (papel = 200-255)
    mask_white = cv2.inRange(thresh, 180, 255)
    
    # 4. Operações morfológicas para UNIR todas as partes do papel
    # Kernel grande para conectar papel impresso + manuscrito
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    mask_white = cv2.dilate(mask_white, kernel_large, iterations=3)
    
    # 5. Remove pequenos ruídos
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # 6. Encontra contornos da área branca
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️ Nenhum contorno encontrado na máscara branca")
        return None
    
    # 7. Pega o maior contorno (deve ser o documento inteiro)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image.shape[0] * image.shape[1]
    
    for cnt in contours[:3]:  # Testa os 3 maiores
        area = cv2.contourArea(cnt)
        
        # Deve ter pelo menos 20% da imagem
        if area < img_area * 0.20:
            continue
        
        # Aproxima para retângulo
        peri = cv2.arcLength(cnt, True)
        
        # Tenta diferentes níveis de aproximação
        for epsilon in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            
            if len(approx) == 4:
                # Verifica se é um quadrilátero razoável
                x, y, w_cnt, h_cnt = cv2.boundingRect(approx)
                aspect_ratio = float(w_cnt) / h_cnt
                
                # Documento vertical (largura < altura)
                if 0.5 < aspect_ratio < 1.2:  # Mais flexível para incluir caligrafia
                    print(f"✓ Contorno aceito: Área={area/img_area*100:.1f}%, Aspect={aspect_ratio:.2f}")
                    return approx.reshape(4, 2).astype("float32")
    
    # FALLBACK: Usa minAreaRect no maior contorno
    print("⚠️ Usando fallback com minAreaRect")
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype("float32")

def enhance_document(image):
    """
    Ajuste de Qualidade: Aumenta contraste e nitidez para dar 'pop' ao texto.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Remoção de Sombra e Normalização
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Ajuste de Contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)
    
    # 3. Nitidez (Unsharp Mask)
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(normalized, 1.8, gaussian, -0.8, 0)
    
    # 4. Ajuste Final de Brilho/Contraste
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def process_image(image):
    """
    Digitalizador que detecta, corrige perspectiva e enquadra documentos.
    INCLUI toda a área do documento, incluindo texto manuscrito.
    """
    orig = image.copy()
    h, w = image.shape[:2]
    
    # 1. Redimensiona para processamento rápido
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ratio = scale
    else:
        image_resized = image.copy()
        ratio = 1.0
    
    # 2. Encontra o contorno do documento (COMPLETO)
    doc_contour_resized = find_document_contour(image_resized)
    
    if doc_contour_resized is None:
        print("❌ Não foi possível detectar o documento")
        return orig 
    
    # 3. Ajusta para escala original
    doc_contour_orig = doc_contour_resized / ratio
    pts = order_points(doc_contour_orig)
    (tl, tr, br, bl) = pts
    
    # 4. Margem pequena (10px) para não cortar nada
    margin = 10
    
    # 5. Calcula dimensões do documento
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB)) + 2 * margin
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + 2 * margin
    
    # 6. Define o destino retangular com margem
    dst = np.array([
        [margin, margin],
        [maxWidth - 1 - margin, margin],
        [maxWidth - 1 - margin, maxHeight - 1 - margin],
        [margin, maxHeight - 1 - margin]
    ], dtype="float32")
    
    # 7. Aplica a transformação de perspectiva
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # 8. Pós-processamento
    warped = enhance_document(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px (Margem: {margin}px)")
    return warped

def draw_contour_debug(image, contour_points, color=(0, 255, 0), thickness=3, show_corners=True):
    """Desenha o contorno detectado para debug."""
    debug_img = image.copy()
    if contour_points is not None:
        pts = np.int32(contour_points)
        cv2.polylines(debug_img, [pts], True, color, thickness)
        
        if show_corners:
            for i, pt in enumerate(pts):
                cv2.circle(debug_img, tuple(pt.flatten()), 10, (0, 0, 255), -1) 
                cv2.putText(debug_img, str(i), tuple(pt.flatten()), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return debug_img

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
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Erro ao carregar: {input_path}")
        return None
    
    print(f"\n📄 Processando: {input_path}")
    print(f"   Dimensões originais: {image.shape[1]}x{image.shape[0]}px")
    
    # Processamento
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Salvo em: {output_path}")
    
    # Mostra debug se solicitado
    if show_debug:
        # Prepara imagem para debug
        temp_h, temp_w = image.shape[:2]
        max_dim_debug = 800
        scale_debug = max_dim_debug / max(temp_h, temp_w)
        image_resized_debug = cv2.resize(image, None, fx=scale_debug, fy=scale_debug, interpolation=cv2.INTER_AREA)
        
        # Detecta e desenha contorno
        doc_contour_for_debug = find_document_contour(image_resized_debug)
        if doc_contour_for_debug is not None:
            doc_contour_for_debug = order_points(doc_contour_for_debug)
            debug_contour_img = draw_contour_debug(image_resized_debug, doc_contour_for_debug)
            cv2.imshow("1. Contorno Detectado (0:TL, 1:TR, 2:BR, 3:BL)", debug_contour_img)
        
        # Mostra resultado
        h_res, w_res = result.shape[:2]
        ratio_res_display = min(800 / w_res, 800 / h_res)
        preview_res = cv2.resize(result, None, fx=ratio_res_display, fy=ratio_res_display, interpolation=cv2.INTER_AREA)
        cv2.imshow("2. Documento Digitalizado (Resultado Final)", preview_res)
        
        print("\n✓ Janelas abertas. Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# ============================================================================
# COMO USAR
# ============================================================================

# Opção 1: Uso simples
# img = cv2.imread('documento.jpg')
# resultado = process_image(img)
# cv2.imwrite('documento_digitalizado.jpg', resultado)

# Opção 2: Uso completo com debug
# scan_document('documento.jpg', 'documento_digitalizado.jpg', show_debug=True)

# Opção 3: Processar múltiplos arquivos
# import glob
# for arquivo in glob.glob('*.jpg'):
#     scan_document(arquivo, f'digitalizado_{arquivo}', show_debug=False)
