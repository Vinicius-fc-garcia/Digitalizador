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
    
    print("Procurando contorno do documento...")
    # --- 2. Encontra o contorno do documento ---
    doc_contour_resized = find_document_contour(image_resized)
    
    if doc_contour_resized is None:
        print("⚠️ Não foi possível detectar um contorno de documento claro. Retornando a imagem original.")
        return orig 
    
    # Ajusta o contorno para a escala original da imagem
    doc_contour_orig = doc_contour_resized / ratio
    
    # --- 3. Ordena os 4 pontos do contorno ---
    pts = order_points(doc_contour_orig)
    
    # --- 4. Calcula as dimensões do documento retificado ---
    (tl, tr, br, bl) = pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
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
    
    # --- 7. Pós-processamento para melhorar qualidade (foco em impressão) ---
    warped = enhance_document_for_print(warped)
    
    print(f"✓ Documento digitalizado: {maxWidth}x{maxHeight}px")
    return warped


def find_document_contour(image):
    """
    ✅ CORRIGIDO: Detecta o contorno do documento (papel) e não da prancheta.
    Remove operações de morfologia que fundiam os contornos.
    Usa RETR_LIST para encontrar todos os contornos e filtra pelo melhor candidato.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur para reduzir ruído de textura e texto
    # Um blur maior é melhor para ignorar o texto dentro da nota
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. Canny para detecção de bordas
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 3. Encontra contornos (TODOS ELES, não só o externo)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # MUDANÇA: RETR_LIST
    
    if not contours:
        print("Canny não encontrou contornos.")
        return None
    
    # 4. Ordena por área (do maior para o menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = image.shape[0] * image.shape[1]
    
    # 5. Procura o melhor contorno retangular entre os maiores
    print(f"Encontrados {len(contours)} contornos. Testando os 20 maiores...")
    for cnt in contours[:20]: # Aumenta para testar mais contornos
        area = cv2.contourArea(cnt)
        
        # Filtro de área (importante)
        # O documento deve ter pelo menos 10% da área da imagem
        if area < img_area * 0.10: 
            break # Como estão ordenados, não há maiores
        
        peri = cv2.arcLength(cnt, True)
        
        # 6. Tenta aproximar para um polígono
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Epsilon 2%
        
        # 7. É um quadrilátero?
        if len(approx) == 4:
            # É convexo? (Evita formas em "C" ou "X")
            if cv2.isContourConvex(approx):
                print(f"✓ Encontrado candidato de 4 lados com área {area}")
                return approx.reshape(4, 2).astype("float32")
            else:
                print(f"Candidato de 4 lados rejeitado (não convexo). Área: {area}")

    print("⚠️ Não foi encontrado nenhum contorno de 4 lados adequado.")
    
    # --- FALLBACK: Se o Canny puro falhar, tenta o método antigo (com morfologia)
    # Isso pode pegar o holder, mas é melhor que nada.
    print("Tentando fallback com morfologia...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    contours_fb, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_fb:
        print("Fallback também falhou.")
        return None
        
    contours_fb = sorted(contours_fb, key=cv2.contourArea, reverse=True)
    cnt_fb = contours_fb[0]
    peri_fb = cv2.arcLength(cnt_fb, True)
    approx_fb = cv2.approxPolyDP(cnt_fb, 0.02 * peri_fb, True)
    
    if len(approx_fb) == 4:
        print("Usando fallback (com morfologia, RETR_EXTERNAL).")
        return approx_fb.reshape(4, 2).astype("float32")

    print("Fallback final falhou. Nenhum contorno de 4 lados encontrado.")
    return None


def order_points(pts):
    """
    Ordena pontos: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def enhance_document_for_print(image):
    """
    ✅ CORRIGIDO: Melhora a qualidade focado em IMPRESSÃO.
    Converte para P&B de alto contraste (Binarização Adaptativa).
    Isso força o texto a ficar preto e o fundo branco.
    """
    print("Aplicando filtro de binarização para impressão...")
    
    # Converte para escala de cinza
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Suaviza levemente para remover ruído antes da binarização
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Binarização adaptativa: calcula o limiar para pequenas regiões.
    # Isso é excelente para texto em fundos com iluminação irregular.
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, # Valor máximo (branco)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Método
        cv2.THRESH_BINARY, # Tipo: texto preto, fundo branco
        15, # Tamanho do bloco (pequeno para capturar detalhes do texto)
        7   # C: Constante subtraída da média
    )
    
    # Converte de volta para BGR (3 canais) para consistência no salvamento
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def draw_contour_debug(image, contour_points, color=(0, 255, 0), thickness=3, show_corners=True):
    """
    Função auxiliar para debug - desenha o contorno detectado na imagem redimensionada.
    """
    debug_img = image.copy()
    if contour_points is not None:
        pts = np.int32(contour_points)
        cv2.polylines(debug_img, [pts], True, color, thickness)
        
        if show_corners:
            for i, pt in enumerate(pts):
                cv2.circle(debug_img, tuple(pt.flatten()), 10, (0, 0, 255), -1) # Canto vermelho
                cv2.putText(debug_img, str(i), tuple(pt.flatten()), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return debug_img


# Exemplo de uso completo
def scan_document(input_path, output_path=None, show_debug=False):
    """
    Função completa para escanear um documento.
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
            # Ordena os pontos de debug para desenhar os números corretamente
            doc_contour_for_debug = order_points(doc_contour_for_debug)
            debug_contour_img = draw_contour_debug(image_resized_debug, doc_contour_for_debug)
            cv2.imshow("Contorno Detectado (Debug)", debug_contour_img)
        else:
            cv2.imshow("Contorno Detectado (Debug)", image_resized_debug) 
            print("Não foi possível desenhar o contorno de debug.")

    # --- Processamento principal ---
    result = process_image(image)
    
    # Salva resultado
    if output_path:
        # Salva como PNG para evitar compressão JPEG em imagem P&B
        # ou usa JPEG com qualidade alta
        if ".png" in output_path.lower():
             cv2.imwrite(output_path, result) 
        else:
             cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Salvo em: {output_path}")
    
    # Mostra debug se solicitado
    if show_debug:
        h_res, w_res = result.shape[:2]
        ratio_res_display = min(800 / w_res, 800 / h_res) 
        preview_res = cv2.resize(result, None, fx=ratio_res_display, fy=ratio_res_display, interpolation=cv2.INTER_AREA)

        cv2.imshow("Original (Reduzido)", image_resized_debug) 
        cv2.imshow("Digitalizado (Resultado Final)", preview_res)
        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


# --- COMO USAR ---

# 1. Salve a imagem original como 'documento.jpg'
# 2. Defina o nome do arquivo de saída
arquivo_entrada = 'documento.jpg' # Use o nome da sua imagem original
arquivo_saida = 'documento_digitalizado_v2.jpg'

# 3. Execute com show_debug=True
scan_document(arquivo_entrada, arquivo_saida, show_debug=True)
