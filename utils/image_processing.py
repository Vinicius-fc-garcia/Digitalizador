import cv2
import numpy as np

def process_image(image):
    """
    Digitalizador que detecta, corrige perspectiva e enquadra documentos.
    Remove completamente as bordas e corrige distorções.
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
        print("⚠️ Não foi possível detectar um contorno de documento claro. Retornando a imagem original.")
        # Se nenhum contorno for detectado, retorne a imagem original
        return orig 
    
    # Ajusta o contorno para a escala original da imagem
    doc_contour_orig = doc_contour_resized / ratio
    
    # --- 3. Ordena os 4 pontos do contorno ---
    pts = order_points(doc_contour_orig)
    
    # --- 4. Calcula as dimensões do documento retificado ---
    (tl, tr, br, bl) = pts
    
    # Calcula a largura máxima
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calcula a altura máxima
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # --- 5. Define o destino retangular para a transformação ---
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # --- 6. Aplica a transformação de perspectiva ---
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # --- 7. Pós-processamento para melhorar qualidade ---
    warped = enhance_document(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px")
    return warped


def find_document_contour(image):
    """
    Detecta o contorno do documento usando múltiplas técnicas.
    Retorna os 4 cantos do documento.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- Pré-processamento avançado ---
    # Suaviza a imagem mantendo as bordas (útil para ruído sem perder detalhes)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17) 
    
    # Melhora contraste com CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = clahe.apply(blurred)

    # Canny para detecção de bordas
    edges = cv2.Canny(blurred, 75, 200, apertureSize=3) # Ajuste de thresholds Canny

    # --- Operações Morfológicas para fechar gaps e limpar ruído ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) # Kernel maior
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3) # Mais iterações
    dilated = cv2.dilate(closed, kernel, iterations=2) # Dilata para conectar ainda mais

    # Encontra contornos na imagem processada
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Ordena por área (do maior para o menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = image.shape[0] * image.shape[1]
    
    # Procura o melhor contorno retangular entre os maiores
    for cnt in contours[:10]: # Aumenta para testar mais contornos
        area = cv2.contourArea(cnt)
        
        # Filtra contornos muito pequenos ou muito grandes (ruído/borda da imagem)
        if area < img_area * 0.15 or area > img_area * 0.95: # Documento deve ocupar entre 15% e 95% da imagem
            continue
        
        peri = cv2.arcLength(cnt, True)
        
        # Tenta diferentes níveis de aproximação para encontrar 4 pontos
        # Variação de epsilon pode ser crucial aqui
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]: 
            approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
            
            if len(approx) == 4:
                # É um quadrilátero com 4 pontos
                # Verificações adicionais: ângulo e convexidade
                if cv2.isContourConvex(approx):
                    # Também podemos verificar a relação de aspecto aqui para evitar contornos muito finos
                    x, y, w_cnt, h_cnt = cv2.boundingRect(approx)
                    aspect_ratio = float(w_cnt)/h_cnt
                    if 0.5 < aspect_ratio < 2.0: # Relação de aspecto razoável para um documento
                        return approx.reshape(4, 2).astype("float32")
                # Se não for convexo ou a proporção não for boa, mas ainda tem 4 pontos
                # Podemos retornar como um fallback, mas a preferência é pelo convexo e bom aspecto
                # return approx.reshape(4, 2).astype("float32") # Comentei para focar no mais robusto
                
    # Fallback: se não encontrou um contorno de 4 pontos bom, 
    # usa o maior contorno e tenta obter um retângulo a partir dele.
    if contours:
        cnt = contours[0] # Maior contorno
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return box.astype("float32")
        
    return None


def order_points(pts):
    """
    Ordena pontos: top-left, top-right, bottom-right, bottom-left.
    Mais robusto contra rotações.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Os pontos com a menor soma (x+y) é o top-left
    # Os pontos com a maior soma (x+y) é o bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Os pontos com a menor diferença (x-y) é o top-right
    # Os pontos com a maior diferença (x-y) é o bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def enhance_document(image):
    """
    Melhora a qualidade do documento digitalizado.
    Remove sombras, aumenta contraste e nitidez.
    Preserva a cor se o original for colorido.
    """
    
    # Se for colorido, processa o canal de Luminosidade (L)
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Remove sombras do canal L
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)) # Kernel ligeiramente maior
        background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        l_normalized = cv2.divide(l, background, scale=255)
        
        # Aumenta contraste do canal L
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10)) # Limite e grid maiores
        l_enhanced = clahe.apply(l_normalized)
        
        # Aplica nitidez no canal L
        gaussian_l = cv2.GaussianBlur(l_enhanced, (0, 0), 2.5) # Sigma maior
        l_sharpened = cv2.addWeighted(l_enhanced, 1.8, gaussian_l, -0.8, 0) # Pesos mais agressivos

        # Ajuste final de brilho/contraste no canal L
        l_final = cv2.convertScaleAbs(l_sharpened, alpha=1.1, beta=10) # Alpha e Beta ajustados
        
        # Mescla canais de volta
        merged = cv2.merge((l_final, a, b))
        final_color = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # --- OPÇÃO PARA IMPRESSÃO (PRETO E BRANCO) ---
        # Descomente as linhas abaixo para um documento P&B de alto contraste
        # print("Convertendo para P&B de alto contraste para impressão.")
        # gray_final = cv2.cvtColor(final_color, cv2.COLOR_BGR2GRAY)
        # _, final_bw = cv2.threshold(gray_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # return cv2.cvtColor(final_bw, cv2.COLOR_GRAY2BGR) # Converte de volta para 3 canais BGR para consistência
            
        return final_color
        
    else:
        # Se já for P&B, usa a lógica de cinza
        gray = image.copy()
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
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
        
        gaussian = cv2.GaussianBlur(normalized, (0, 0), 2.5)
        sharpened = cv2.addWeighted(normalized, 1.8, gaussian, -0.8, 0)
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def draw_contour_debug(image, contour_points, color=(0, 255, 0), thickness=3, show_corners=True):
    """
    Função auxiliar para debug - desenha o contorno detectado na imagem redimensionada.
    """
    debug_img = image.copy()
    if contour_points is not None:
        # Garante que os pontos são inteiros para o drawing
        pts = np.int32(contour_points)
        cv2.polylines(debug_img, [pts], True, color, thickness)
        
        if show_corners:
            # Desenha círculos nos cantos e numera
            for i, pt in enumerate(pts):
                cv2.circle(debug_img, tuple(pt.flatten()), 10, (0, 0, 255), -1) # Canto vermelho
                # Adiciona texto para identificar os cantos (0: TL, 1: TR, 2: BR, 3: BL)
                cv2.putText(debug_img, str(i), tuple(pt.flatten()), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return debug_img


# Exemplo de uso completo
def scan_document(input_path, output_path=None, show_debug=False):
    """
    Função completa para escanear um documento.
    
    Args:
        input_path: Caminho da imagem de entrada
        output_path: Caminho para salvar (opcional)
        show_debug: Se True, mostra imagem com contorno detectado e resultado final
    
    Returns:
        Imagem digitalizada
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Erro ao carregar: {input_path}")
        return None
    
    print(f"📄 Processando: {input_path}")
    print(f"    Dimensões originais: {image.shape[1]}x{image.shape[0]}px")

    # --- Pré-processamento e detecção de contorno para DEBUG ---
    # Redimensiona para encontrar o contorno (sem alterar a original ainda)
    temp_h, temp_w = image.shape[:2]
    max_dim_debug = 800
    if max(temp_h, temp_w) > max_dim_debug:
        scale_debug = max_dim_debug / max(temp_h, temp_w)
        image_resized_debug = cv2.resize(image, None, fx=scale_debug, fy=scale_debug, interpolation=cv2.INTER_AREA)
        ratio_debug = scale_debug
    else:
        image_resized_debug = image.copy()
        ratio_debug = 1.0

    if show_debug:
        # Tenta encontrar o contorno apenas para visualização de debug
        doc_contour_for_debug = find_document_contour(image_resized_debug)
        if doc_contour_for_debug is not None:
            debug_contour_img = draw_contour_debug(image_resized_debug, doc_contour_for_debug)
            cv2.imshow("Contorno Detectado (Debug)", debug_contour_img)
        else:
            cv2.imshow("Contorno Detectado (Debug)", image_resized_debug) # Mostra a imagem redimensionada se não achou contorno
            print("Não foi possível desenhar o contorno de debug.")

    # --- Processamento principal ---
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        # cv2.IMWRITE_JPEG_QUALITY, 95 garante boa qualidade para JPEG
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        print(f"💾 Salvo em: {output_path}")
    
    # Mostra debug se solicitado
    if show_debug:
        h_res, w_res = result.shape[:2]
        # Redimensiona o resultado para caber na tela mantendo a proporção
        ratio_res_display = min(800 / w_res, 800 / h_res) # Adaptação para caber em 800x800
        preview_res = cv2.resize(result, None, fx=ratio_res_display, fy=ratio_res_display, interpolation=cv2.INTER_AREA)

        cv2.imshow("Original (Reduzido)", image_resized_debug) # Mostra original reduzido
        cv2.imshow("Digitalizado (Resultado Final)", preview_res)
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# --- COMO USAR ---

# 1. Salve a imagem que você enviou como 'documento.jpg' no mesmo diretório
# 2. Descomente a linha abaixo para executar
arquivo_entrada = 'documento.jpg' # Certifique-se que o nome do arquivo está correto
arquivo_saida = 'documento_digitalizado.jpg'

# Chame a função scan_document com show_debug=True para ver o processo
scan_document(arquivo_entrada, arquivo_saida, show_debug=True)

# Ou, se preferir apenas processar e salvar (sem debug visual):
# scan_document(arquivo_entrada, arquivo_saida, show_debug=False)
