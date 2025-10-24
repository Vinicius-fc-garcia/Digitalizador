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
    ✅ CORREÇÃO: Ajustes para incluir a área de caligrafia.
    Remove filtros de proporção e ajusta Canny/Blur/Epsilon.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur um pouco mais leve (5x5) para tentar pegar as bordas da caligrafia
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny para detecção de bordas (thresholds mais abertos para caligrafia)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3) # Ajustado para ser mais sensível
    
    # 3. Encontra contornos (TODOS ELES)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    
    if not contours:
        return None
    
    # 4. Ordena por área (do maior para o menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = image.shape[0] * image.shape[1]
    
    # 5. Procura o melhor contorno retangular entre os maiores
    for cnt in contours[:20]: 
        area = cv2.contourArea(cnt)
        
        if area < img_area * 0.10: 
            break
        
        peri = cv2.arcLength(cnt, True)
        
        # 6. Tenta aproximar para um polígono (epsilon aumentado para 4-5% - mais flexível)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) # Aumentado para 4%
        
        # 7. É um quadrilátero convexo? (Remover filtro de aspecto aqui)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            print(f"✓ Contorno aceito. Área: {area}")
            return approx.reshape(4, 2).astype("float32")

    print("⚠️ Não foi encontrado nenhum contorno de 4 lados adequado. Tentando Fallback.")
    
    # --- FALLBACK: Usar Morfologia, que pega o contorno maior ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    contours_fb, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_fb: return None
        
    contours_fb = sorted(contours_fb, key=cv2.contourArea, reverse=True)
    cnt_fb = contours_fb[0]
    peri_fb = cv2.arcLength(cnt_fb, True)
    approx_fb = cv2.approxPolyDP(cnt_fb, 0.02 * peri_fb, True)
    
    if len(approx_fb) == 4:
        print("Usando contorno Fallback (com morfologia).")
        return approx_fb.reshape(4, 2).astype("float32")

    return None

def enhance_document(image):
    """
    Ajuste de Qualidade: Aumenta contraste e nitidez para dar 'pop' ao texto.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # --- 1. Remoção de Sombra e Normalização ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    # --- 2. Ajuste de Contraste (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)
    
    # --- 3. Nitidez (Unsharp Mask) ---
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(normalized, 1.8, gaussian, -0.8, 0)
    
    # --- 4. Ajuste Final de Brilho/Contraste ---
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def process_image(image):
    """
    Digitalizador que detecta, corrige perspectiva e enquadra documentos.
    CORREÇÃO: Margem ajustada para evitar corte lateral enquanto inclui a caligrafia.
    """
    orig = image.copy()
    h, w = image.shape[:2]
    
    # --- 1. Redimensiona para processamento rápido ---
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ratio = scale
    else:
        image_resized = image.copy()
        ratio = 1.0
    
    # --- 2. Encontra o contorno do documento ---
    doc_contour_resized = find_document_contour(image_resized)
    
    if doc_contour_resized is None:
        return orig 
    
    doc_contour_orig = doc_contour_resized / ratio
    pts = order_points(doc_contour_orig)
    (tl, tr, br, bl) = pts
    
    # MARGEM AJUSTADA: 25px é um bom meio-termo
    margin = 25 
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB)) + 2 * margin
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + 2 * margin
    
    # --- 5. Define o destino retangular para a transformação com a margem ---
    dst = np.array([
        [margin, margin],
        [maxWidth - 1 - margin, margin],
        [maxWidth - 1 - margin, maxHeight - 1 - margin],
        [margin, maxHeight - 1 - margin]
    ], dtype="float32")
    
    # --- 6. Aplica a transformação de perspectiva ---
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # --- 7. Pós-processamento ---
    warped = enhance_document(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px (Margem de {margin}px aplicada)")
    return warped

# --- Funções de Debug (Manter) ---
def draw_contour_debug(image, contour_points, color=(0, 255, 0), thickness=3, show_corners=True):
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
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Erro ao carregar: {input_path}")
        return None
    
    print(f"📄 Processando: {input_path}")
    
    result = process_image(image)
    
    if output_path:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Salvo em: {output_path}")
    
    if show_debug:
        temp_h, temp_w = image.shape[:2]
        max_dim_debug = 800
        scale_debug = max_dim_debug / max(temp_h, temp_w)
        image_resized_debug = cv2.resize(image, None, fx=scale_debug, fy=scale_debug, interpolation=cv2.INTER_AREA)
        
        doc_contour_for_debug = find_document_contour(image_resized_debug)
        if doc_contour_for_debug is not None:
            doc_contour_for_debug = order_points(doc_contour_for_debug)
            debug_contour_img = draw_contour_debug(image_resized_debug, doc_contour_for_debug)
            cv2.imshow("Contorno Detectado (Debug - 0:TL, 1:TR, 2:BR, 3:BL)", debug_contour_img)
        
        h_res, w_res = result.shape[:2]
        ratio_res_display = min(800 / w_res, 800 / h_res)
        preview_res = cv2.resize(result, None, fx=ratio_res_display, fy=ratio_res_display, interpolation=cv2.INTER_AREA)

        cv2.imshow("Digitalizado (Resultado Final)", preview_res)
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# --- COMO USAR ---
# 1. Certifique-se de que a imagem original está salva como 'documento.jpg'
arquivo_entrada = 'documento.jpg' 
arquivo_saida = 'documento_digitalizado_caligrafia_final.jpg'

# 2. Execute com show_debug=True para ver o contorno detectado
scan_document(arquivo_entrada, arquivo_saida, show_debug=True)
