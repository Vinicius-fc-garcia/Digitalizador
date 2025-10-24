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
    VERSÃO FINAL: Detecta o documento COMPLETO (incluindo manuscrito)
    e garante que encontre os 4 cantos corretamente para correção de perspectiva.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    
    # 1. Blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. DETECÇÃO DE ÁREA BRANCA (papel) - método que pega tudo
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_white = cv2.inRange(thresh, 180, 255)
    
    # 3. Morfologia para UNIR documento impresso + manuscrito
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    mask_white = cv2.dilate(mask_white, kernel_large, iterations=3)
    
    # 4. CANNY nas bordas externas (para encontrar os cantos precisos)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 5. Combina máscara branca + bordas
    # Isso garante que pegamos a área completa MAS com bordas bem definidas
    combined = cv2.bitwise_or(mask_white, edges)
    
    # 6. Dilata o resultado combinado para garantir contorno fechado
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    combined = cv2.dilate(combined, kernel_dilate, iterations=2)
    
    # 7. Encontra contornos
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️ Nenhum contorno encontrado")
        return None
    
    # 8. Ordena por área
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = h * w
    
    # 9. Procura o melhor contorno de 4 lados
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        
        # Filtro: deve ter pelo menos 20% da imagem
        if area < img_area * 0.20:
            continue
        
        peri = cv2.arcLength(cnt, True)
        
        # Tenta diferentes níveis de aproximação
        for epsilon in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            
            if len(approx) == 4:
                # Verifica se é um quadrilátero válido
                x, y, w_cnt, h_cnt = cv2.boundingRect(approx)
                aspect_ratio = float(w_cnt) / h_cnt
                
                # Documento vertical: largura < altura (tolerância flexível)
                if 0.4 < aspect_ratio < 1.3:
                    # Verifica se o contorno é grande o suficiente
                    contour_area_ratio = area / img_area
                    if contour_area_ratio > 0.25:
                        print(f"✓ Contorno 4 lados: Área={contour_area_ratio*100:.1f}%, Aspect={aspect_ratio:.2f}, Epsilon={epsilon}")
                        return approx.reshape(4, 2).astype("float32")
    
    # FALLBACK MELHORADO: minAreaRect no maior contorno
    print("⚠️ Não encontrou 4 lados perfeitos. Usando minAreaRect...")
    cnt = contours[0]
    
    # Usa minAreaRect para encontrar o retângulo mínimo
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    
    # Verifica se o retângulo está muito inclinado
    # Se sim, tenta ajustar para os eixos da imagem
    (center, (width, height), angle) = rect
    
    print(f"   MinAreaRect: Ângulo={angle:.1f}°, {width:.0f}x{height:.0f}px")
    
    return box.astype("float32")

def enhance_document(image):
    """
    Melhora qualidade: remove sombras, aumenta contraste e nitidez.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Remove sombras
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. CLAHE para contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)
    
    # 3. Sharpening (Unsharp Mask)
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(normalized, 1.8, gaussian, -0.8, 0)
    
    # 4. Ajuste final
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def process_image(image):
    """
    Digitalizador COMPLETO: detecta todo o documento (incluindo manuscrito)
    e aplica correção de perspectiva precisa.
    """
    orig = image.copy()
    h, w = image.shape[:2]
    
    # 1. Redimensiona para processamento
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ratio = scale
    else:
        image_resized = image.copy()
        ratio = 1.0
    
    # 2. Detecta o contorno do documento completo
    doc_contour_resized = find_document_contour(image_resized)
    
    if doc_contour_resized is None:
        print("❌ Falha na detecção do documento")
        return orig
    
    # 3. Ajusta para escala original
    doc_contour_orig = doc_contour_resized / ratio
    
    # 4. Ordena os pontos corretamente
    pts = order_points(doc_contour_orig)
    (tl, tr, br, bl) = pts
    
    # 5. Calcula dimensões do documento retificado
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 6. Adiciona margem mínima (5px apenas para segurança)
    margin = 5
    maxWidth += 2 * margin
    maxHeight += 2 * margin
    
    # 7. Define pontos de destino (retângulo perfeito)
    dst = np.array([
        [margin, margin],
        [maxWidth - 1 - margin, margin],
        [maxWidth - 1 - margin, maxHeight - 1 - margin],
        [margin, maxHeight - 1 - margin]
    ], dtype="float32")
    
    # 8. Calcula e aplica transformação de perspectiva
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # 9. Pós-processamento para melhorar qualidade
    warped = enhance_document(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px")
    print(f"   Cantos detectados: TL{tuple(tl.astype(int))}, TR{tuple(tr.astype(int))}, BR{tuple(br.astype(int))}, BL{tuple(bl.astype(int))}")
    
    return warped

def draw_contour_debug(image, contour_points, color=(0, 255, 0), thickness=3, show_corners=True):
    """Desenha o contorno detectado para debug visual."""
    debug_img = image.copy()
    if contour_points is not None:
        pts = np.int32(contour_points)
        
        # Desenha o contorno
        cv2.polylines(debug_img, [pts], True, color, thickness)
        
        if show_corners:
            # Desenha os cantos numerados
            labels = ['TL', 'TR', 'BR', 'BL']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, (pt, label, col) in enumerate(zip(pts, labels, colors)):
                pt_tuple = tuple(pt.flatten())
                cv2.circle(debug_img, pt_tuple, 12, col, -1)
                cv2.circle(debug_img, pt_tuple, 15, (255, 255, 255), 2)
                cv2.putText(debug_img, label, (pt_tuple[0]-15, pt_tuple[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return debug_img

def scan_document(input_path, output_path=None, show_debug=False):
    """
    Função completa para digitalizar documentos.
    
    Args:
        input_path: Caminho da imagem de entrada
        output_path: Caminho para salvar resultado (opcional)
        show_debug: Se True, exibe janelas de debug
    
    Returns:
        Imagem digitalizada (numpy array)
    """
    # Carrega imagem
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Erro ao carregar: {input_path}")
        return None
    
    print(f"\n📄 Processando: {input_path}")
    print(f"   Dimensões originais: {image.shape[1]}x{image.shape[0]}px")
    
    # Processa
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        success = cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            print(f"💾 Salvo em: {output_path}")
        else:
            print(f"❌ Erro ao salvar em: {output_path}")
    
    # Debug visual
    if show_debug:
        # Prepara imagem redimensionada para debug
        h_orig, w_orig = image.shape[:2]
        max_debug = 900
        scale_debug = min(max_debug / w_orig, max_debug / h_orig)
        img_debug = cv2.resize(image, None, fx=scale_debug, fy=scale_debug, interpolation=cv2.INTER_AREA)
        
        # Detecta e desenha contorno
        contour_debug = find_document_contour(img_debug)
        if contour_debug is not None:
            contour_ordered = order_points(contour_debug)
            img_with_contour = draw_contour_debug(img_debug, contour_ordered)
            cv2.imshow("1. Contorno Detectado", img_with_contour)
        
        # Mostra resultado
        h_res, w_res = result.shape[:2]
        scale_result = min(max_debug / w_res, max_debug / h_res)
        result_display = cv2.resize(result, None, fx=scale_result, fy=scale_result, interpolation=cv2.INTER_AREA)
        cv2.imshow("2. Documento Digitalizado", result_display)
        
        print("\n✓ Janelas de debug abertas. Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

# Uso básico:
# img = cv2.imread('documento.jpg')
# resultado = process_image(img)
# cv2.imwrite('digitalizado.jpg', resultado)

# Uso completo com debug:
# scan_document('documento.jpg', 'digitalizado.jpg', show_debug=True)

# Processar múltiplos arquivos:
# import glob
# for arquivo in glob.glob('*.jpg'):
#     nome_saida = arquivo.replace('.jpg', '_digitalizado.jpg')
#     scan_document(arquivo, nome_saida)
