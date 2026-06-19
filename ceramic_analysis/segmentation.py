# -*- coding: utf-8 -*-
"""
Módulo de segmentação e detecção de contornos para corpos de prova cerâmicos.

Implementa múltiplas estratégias de segmentação para lidar com diferentes
condições de contraste:
- background_sub: Subtração de fundo (recomendada — funciona para qualquer cor)
- lab: Espaço de cor LAB (bom para baixo contraste de luminância)
- otsu: Threshold automático de Otsu (bom para alto contraste)
- adaptive: Threshold adaptativo (iluminação desigual)
- auto: Tenta cada estratégia e seleciona a melhor

Uso:
    from segmentation import segment, SegmentationError

    # Com background
    results = segment(gray, color_img, "sample_01", background=bg_img)

    # Sem background (usa LAB ou Otsu)
    results = segment(gray, color_img, "sample_01")
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np

import config
import preprocessing

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Exceção quando nenhum contorno válido é encontrado."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Estratégias de segmentação
# ──────────────────────────────────────────────────────────────────────────────

def segment_background_sub(
    image_color: np.ndarray,
    background_color: np.ndarray,
    threshold: int = None
) -> np.ndarray:
    """
    Segmentação por subtração de fundo.

    Calcula a diferença absoluta entre a imagem com a peça e a foto da
    base vazia (grade no MDF sem corpo de prova). Funciona para qualquer
    cor de argila, inclusive marrom sobre MDF marrom.

    Args:
        image_color: Imagem com o corpo de prova (BGR, uint8).
        background_color: Imagem da base vazia (BGR, uint8).
        threshold: Limiar de diferença (0-255).
                   ↑ Aumentar se fundo ruidoso gera falsos positivos.
                   ↓ Diminuir se partes da peça são cortadas.
                   Default: config.BG_SUB_THRESHOLD (25)

    Returns:
        Máscara binária (uint8): 255 = objeto, 0 = fundo.
    """
    threshold = threshold if threshold is not None else config.BG_SUB_THRESHOLD

    if getattr(config, "BG_SUB_NORMALIZE_BRIGHTNESS", False):
        background_color = preprocessing.normalize_brightness(background_color, image_color)

    # Diferença absoluta por canal BGR
    diff = cv2.absdiff(image_color, background_color)

    # Máximo entre os 3 canais — captura diferenças em qualquer canal
    # (mais sensível que converter para cinza e depois diferenciar)
    gray_diff = np.max(diff, axis=2)

    # Binarizar
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    return mask


def segment_lab(image_color: np.ndarray) -> np.ndarray:
    """
    Segmentação via espaço de cor LAB.

    O espaço LAB separa luminância (L) de cromaticidade (A, B):
    - Canal A: verde ←→ vermelho
    - Canal B: azul ←→ amarelo

    Mesmo quando argila e MDF têm luminância similar, as diferenças
    cromáticas nos canais A ou B permitem a segmentação.

    O canal com maior variância (melhor separação peça/fundo) é escolhido
    automaticamente.

    Args:
        image_color: Imagem BGR (uint8).

    Returns:
        Máscara binária (uint8): 255 = objeto, 0 = fundo.
    """
    lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Escolher canal com maior variância (melhor separação bimodal)
    var_a = np.var(a_ch)
    var_b = np.var(b_ch)
    var_l = np.var(l_ch)

    if var_a >= var_b and var_a >= var_l:
        channel = a_ch
        channel_name = "A"
    elif var_b >= var_a and var_b >= var_l:
        channel = b_ch
        channel_name = "B"
    else:
        channel = l_ch
        channel_name = "L"

    logger.debug(
        f"  LAB: var(L)={var_l:.1f}, var(A)={var_a:.1f}, var(B)={var_b:.1f} "
        f"→ usando canal {channel_name}"
    )

    # Otsu no canal escolhido
    _, mask = cv2.threshold(channel, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


def segment_yellow(image_color: np.ndarray) -> np.ndarray:
    """
    Segmentação específica para peças amarelas.

    Usa o espaço de cores HSV para filtrar tons de amarelo/creme.
    Limiares ajustados para ignorar o fundo MDF (via Hue >= 22) e capturar tons mais claros.
    """
    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    
    # Amarelo/Creme: Hue entre 22 e 45, Saturation > 30, Value > 100
    lower_yellow = np.array([22, 30, 100])
    upper_yellow = np.array([45, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask


def segment_lab_b(image_color: np.ndarray) -> np.ndarray:
    """
    Segmentação específica para peças amarelas/creme usando canal B do LAB.
    
    O canal B do espaço LAB separa Azul (-) de Amarelo (+).
    O MDF possui B médio em torno de 145, enquanto a peça amarela tem B > 149.
    Um limiar dinâmico baseado na média e desvio padrão é muito eficaz.
    """
    lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2LAB)
    b_ch = lab[:, :, 2]
    
    # Criar máscara para ignorar pixels pretos (gerados pelo grid_mask)
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    valid_pixels_mask = gray > 0
    
    if not np.any(valid_pixels_mask):
        b_mean = np.mean(b_ch)
        b_std = np.std(b_ch)
    else:
        b_mean = np.mean(b_ch[valid_pixels_mask])
        b_std = np.std(b_ch[valid_pixels_mask])
        
    factor = getattr(config, "LAB_B_SIGMA_FACTOR", 1.5)
    b_thresh = b_mean + factor * b_std
    
    logger.debug(f"  LAB-B: threshold={b_thresh:.1f} (mean={b_mean:.1f}, std={b_std:.1f}, factor={factor})")
    
    _, mask = cv2.threshold(b_ch, b_thresh, 255, cv2.THRESH_BINARY)
    
    # Garantir que a máscara não inclua as áreas pretas
    mask = cv2.bitwise_and(mask, mask, mask=valid_pixels_mask.astype(np.uint8) * 255)
    
    return mask



def segment_otsu(gray_blurred: np.ndarray) -> np.ndarray:
    """
    Segmentação por threshold automático de Otsu.

    Calcula o limiar ótimo que minimiza a variância intra-classe,
    assumindo uma distribuição bimodal (pico do fundo + pico do objeto).

    Funciona melhor com alto contraste (ex: caulim branco sobre MDF).

    THRESH_BINARY_INV: objeto escuro → branco na máscara.
    Se a peça for mais clara que o fundo, o pós-processamento inverte.

    Args:
        gray_blurred: Imagem em escala de cinza com filtro gaussiano (uint8).

    Returns:
        Máscara binária (uint8).
    """
    if config.MANUAL_THRESHOLD is not None:
        # Override manual — útil para ajuste fino em condições específicas
        thresh_val = config.MANUAL_THRESHOLD
        _, mask = cv2.threshold(gray_blurred, thresh_val, 255,
                                cv2.THRESH_BINARY_INV)
        logger.debug(f"  Threshold manual: {thresh_val}")
    else:
        thresh_val, mask = cv2.threshold(gray_blurred, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        logger.debug(f"  Threshold Otsu automático: {thresh_val}")

    return mask


def segment_otsu_dark(gray_blurred: np.ndarray) -> np.ndarray:
    """
    Segmentação por Otsu considerando apenas pixels escuros (< 160).
    Exclui pixels muito claros da calibração (bloco branco/grade) para
    evitar que enviesem o limiar de Otsu.
    """
    # Selecionar pixels < 160
    pixels = gray_blurred[gray_blurred < 160]
    
    if len(pixels) == 0:
        # Fallback para Otsu padrão se não houver pixels escuros
        _, mask = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask
        
    # Calcular histograma para pixels < 160 (limite 160)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    
    # Algoritmo de Otsu manual no histograma
    total = len(pixels)
    current_max = -1.0
    thresh_val = 0
    
    sum_total = np.sum(np.arange(256) * hist)
    sum_b = 0.0
    w_b = 0.0
    
    for t in range(160):  # Apenas até 160, pois os outros bins são 0
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
            
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        
        # Variância entre classes
        var_between = w_b * w_f * ((m_b - m_f) ** 2)
        
        if var_between > current_max:
            current_max = var_between
            thresh_val = t
            
    logger.debug(f"  Otsu Dark: limiar calculado = {thresh_val}")
    
    # Binarizar a imagem usando o limiar encontrado
    _, mask = cv2.threshold(gray_blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return mask


def segment_adaptive(gray_blurred: np.ndarray) -> np.ndarray:
    """
    Segmentação por threshold adaptativo gaussiano.

    Calcula o limiar localmente por blocos da imagem, compensando
    variações de iluminação na bancada.

    Args:
        gray_blurred: Imagem em escala de cinza (uint8).

    Returns:
        Máscara binária (uint8).
    """
    mask = cv2.adaptiveThreshold(
        gray_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.ADAPTIVE_BLOCK_SIZE,  # Tamanho do bloco (ímpar)
        config.ADAPTIVE_C            # Constante subtraída da média
    )
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Avaliação de qualidade da máscara
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_mask_quality(mask: np.ndarray) -> dict:
    """
    Avalia a qualidade de uma máscara binária de segmentação.

    Critérios:
    - Proporção de pixels brancos (objeto) deve estar entre 1% e 80%
    - Deve haver pelo menos 1 contorno com área > MIN_CONTOUR_AREA_PX
    - Compacidade do maior contorno (4π·area/perimeter²) razoável

    Args:
        mask: Máscara binária (uint8).

    Returns:
        Dict com métricas de qualidade e flag 'is_good'.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = cv2.countNonZero(mask)
    proportion = white_pixels / total_pixels

    # Encontrar contornos para avaliar forma
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtrar por área mínima, circularidade e aspect_ratio
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= config.MIN_CONTOUR_AREA_PX:
            _, _, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity >= 0.05 and aspect_ratio <= 5.0:
                # Rejeitar contornos que cobrem a maior parte de ambas as dimensoes da imagem (MDF de fundo)
                # ou que excedam o limite de área máxima (proporcional ou absoluto em pixels para testes)
                if (w < 0.8 * mask.shape[1] or h < 0.8 * mask.shape[0]) and (area < config.MAX_CONTOUR_AREA_PROPORTION * total_pixels or area < 150000):
                    valid_contours.append(c)

    # Calcular compacidade do maior contorno
    compactness = 0.0
    if valid_contours:
        largest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter * perimeter)

    # Critérios de qualidade
    proportion_ok = 0.01 <= proportion <= 0.80
    has_contours = len(valid_contours) > 0
    compactness_ok = compactness > 0.05  # Consistente com filtro de circularidade

    is_good = proportion_ok and has_contours and compactness_ok

    return {
        "proportion": proportion,
        "n_contours": len(valid_contours),
        "compactness": compactness,
        "is_good": is_good
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pós-processamento e extração de métricas
# ──────────────────────────────────────────────────────────────────────────────

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Aplica fechamento morfológico para eliminar lacunas nas bordas.

    Usa kernel elíptico (bordas mais naturais que retangular) com
    MORPH_CLOSE = dilatação seguida de erosão.

    Isso fecha pequenas lacunas causadas por reflexos na superfície
    da argila úmida, sem alterar significativamente o tamanho do contorno.

    Args:
        mask: Máscara binária (uint8).

    Returns:
        Máscara binária pós-processada.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE
    )
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel,
        iterations=config.MORPH_ITERATIONS
    )
    return closed


def extract_contour_metrics(contour: np.ndarray) -> dict:
    """
    Extrai métricas geométricas de um contorno.

    Calcula múltiplas representações para suportar geometrias variadas:
    - Bounding box: dimensão geral rápida
    - Retângulo mínimo: prismas/retângulos rotacionados
    - Elipse ajustada: cilindros (vista superior circular/elíptica)
    - Área e perímetro: todas as geometrias
    - Casco convexo: formas complexas
    - Circularidade: classificação automática (1.0 = círculo perfeito)

    Args:
        contour: Contorno OpenCV (array Nx1x2).

    Returns:
        Dict com todas as métricas em pixels.
    """
    # Bounding box alinhado aos eixos
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

    # Retângulo de área mínima (rotacionado)
    min_rect = cv2.minAreaRect(contour)
    min_rect_center, (min_rect_w, min_rect_h), min_rect_angle = min_rect

    # Área e perímetro
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    # Circularidade: 1.0 = círculo perfeito, 0.0 = forma muito irregular
    circularity = 0.0
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

    # Casco convexo
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    # Solidez: razão entre área do contorno e área do casco convexo
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Elipse ajustada (requer pelo menos 5 pontos)
    ellipse_major = 0.0
    ellipse_minor = 0.0
    ellipse_angle = 0.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipse_center, (ellipse_minor, ellipse_major), ellipse_angle = ellipse
        except cv2.error:
            logger.debug("  Não foi possível ajustar elipse ao contorno")

    # Vértices e ângulos internos (se for retangular)
    corner_angle_0 = 0.0
    corner_angle_1 = 0.0
    corner_angle_2 = 0.0
    corner_angle_3 = 0.0
    corners_px = []
    corner_angles = []

    if circularity <= 0.80:
        try:
            # 1. Obter os 4 cantos do min_rect
            box = cv2.boxPoints(min_rect)
            cx, cy = min_rect_center
            # Ordenar anti-horário
            box_angles = np.arctan2(box[:, 1] - cy, box[:, 0] - cx)
            sort_idx = np.argsort(box_angles)
            box_sorted = box[sort_idx]
            
            # 2. Projetar no contorno para achar vértices físicos
            for pt in box_sorted:
                dists = np.linalg.norm(contour[:, 0] - pt, axis=1)
                idx = np.argmin(dists)
                corners_px.append(contour[idx, 0])
                
            # 3. Calcular os ângulos internos
            n = len(corners_px)
            for i in range(n):
                pt_curr = corners_px[i]
                pt_prev = corners_px[(i - 1) % n]
                pt_next = corners_px[(i + 1) % n]
                
                u = pt_prev - pt_curr
                v = pt_next - pt_curr
                
                dot_prod = np.dot(u, v)
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                
                if norm_u > 0 and norm_v > 0:
                    cos_theta = np.clip(dot_prod / (norm_u * norm_v), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))
                    corner_angles.append(angle)
                else:
                    corner_angles.append(90.0)
            
            if len(corner_angles) == 4:
                corner_angle_0 = corner_angles[0]
                corner_angle_1 = corner_angles[1]
                corner_angle_2 = corner_angles[2]
                corner_angle_3 = corner_angles[3]
        except Exception as e:
            logger.debug(f"Erro ao calcular ângulos dos vértices: {e}")

    metrics = {
        # Bounding box
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        # Retângulo mínimo
        "min_rect_w": min_rect_w,
        "min_rect_h": min_rect_h,
        "min_rect_angle": min_rect_angle,
        "min_rect_center_x": min_rect_center[0],
        "min_rect_center_y": min_rect_center[1],
        # Área e perímetro
        "area_px": area,
        "perimeter_px": perimeter,
        # Forma
        "circularity": circularity,
        "solidity": solidity,
        # Elipse
        "ellipse_major_px": ellipse_major,
        "ellipse_minor_px": ellipse_minor,
        "ellipse_angle": ellipse_angle,
        # Vértices e ângulos
        "corners_px": [list(pt) for pt in corners_px] if corners_px else None,
        "corner_angles": corner_angles if corner_angles else None,
        "corner_angle_0": corner_angle_0,
        "corner_angle_1": corner_angle_1,
        "corner_angle_2": corner_angle_2,
        "corner_angle_3": corner_angle_3,
        # Contorno original (para desenho e conversão)
        "contour": contour,
        "hull": hull,
    }

    # Calcular seções transversais (3 seções em cada eixo: 10%, 50%, 90%)
    cross_sections = compute_cross_sections(contour, min_rect)
    if cross_sections:
        metrics.update(cross_sections)

    return metrics


def compute_cross_sections(
    contour: np.ndarray,
    min_rect: tuple,
    positions: list = None
) -> dict:
    """
    Calcula medições em seções transversais ao longo dos eixos maior e menor.

    Traça retas perpendiculares ao eixo em posições relativas (ex: 10%, 50%, 90%)
    e mede a distância entre as interseções com o contorno.

    Args:
        contour: Contorno OpenCV (array Nx1x2).
        min_rect: Resultado de cv2.minAreaRect(contour) — (center, (w, h), angle).
        positions: Lista de posições relativas (0.0 a 1.0). Default: [0.10, 0.50, 0.90].

    Returns:
        Dict com as medidas em pixels:
            - cross_width_10pct_px, cross_width_50pct_px, cross_width_90pct_px
            - cross_length_10pct_px, cross_length_50pct_px, cross_length_90pct_px
            - cross_width_pts (lista de pares de pontos para desenho)
            - cross_length_pts (lista de pares de pontos para desenho)
        Retorna None se o contorno for insuficiente.
    """
    if positions is None:
        positions = [0.10, 0.50, 0.90]

    center, (rect_w, rect_h), angle = min_rect

    # Garantir que o eixo maior é o "comprimento" e o menor é a "largura"
    if rect_w >= rect_h:
        major_len = rect_w
        minor_len = rect_h
        theta = np.radians(angle)
    else:
        major_len = rect_h
        minor_len = rect_w
        theta = np.radians(angle + 90)

    # Vetores unitários do eixo maior e menor
    u_major = np.array([np.cos(theta), np.sin(theta)])
    u_minor = np.array([-np.sin(theta), np.cos(theta)])
    center_pt = np.array(center)

    # Converter contorno para array 2D
    pts = contour.reshape(-1, 2).astype(np.float64)

    result = {}
    width_pts_list = []
    length_pts_list = []

    # Seções transversais ao longo do eixo MAIOR (medem LARGURA)
    for pos in positions:
        t = pos - 0.5  # -0.4, 0.0, 0.4
        origin = center_pt + t * major_len * u_major

        # Encontrar interseções do contorno com a reta perpendicular ao eixo maior
        intersections = _find_contour_line_intersections(pts, origin, u_minor)

        if len(intersections) >= 2:
            # Pegar os 2 pontos mais distantes (extremos)
            intersections = sorted(intersections, key=lambda p: np.dot(p - origin, u_minor))
            p1 = intersections[0]
            p2 = intersections[-1]
            width_px = np.linalg.norm(p2 - p1)
            pos_pct = int(pos * 100)
            result[f"cross_width_{pos_pct}pct_px"] = width_px
            width_pts_list.append((p1.tolist(), p2.tolist(), pos))
        else:
            pos_pct = int(pos * 100)
            result[f"cross_width_{pos_pct}pct_px"] = 0.0

    # Seções transversais ao longo do eixo MENOR (medem COMPRIMENTO)
    for pos in positions:
        t = pos - 0.5
        origin = center_pt + t * minor_len * u_minor

        intersections = _find_contour_line_intersections(pts, origin, u_major)

        if len(intersections) >= 2:
            intersections = sorted(intersections, key=lambda p: np.dot(p - origin, u_major))
            p1 = intersections[0]
            p2 = intersections[-1]
            length_px = np.linalg.norm(p2 - p1)
            pos_pct = int(pos * 100)
            result[f"cross_length_{pos_pct}pct_px"] = length_px
            length_pts_list.append((p1.tolist(), p2.tolist(), pos))
        else:
            pos_pct = int(pos * 100)
            result[f"cross_length_{pos_pct}pct_px"] = 0.0

    result["cross_width_pts"] = width_pts_list
    result["cross_length_pts"] = length_pts_list

    return result if any(v > 0 for k, v in result.items() if k.endswith("_px")) else None


def _find_contour_line_intersections(
    pts: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray
) -> list:
    """
    Encontra pontos de interseção entre um contorno (polígono) e uma reta infinita.

    A reta é definida por origin + t * direction.
    Cada aresta do contorno é testada para interseção.

    Args:
        pts: Array Nx2 de pontos do contorno.
        origin: Ponto de origem da reta (2D).
        direction: Vetor direção da reta (2D).

    Returns:
        Lista de np.array (pontos de interseção).
    """
    intersections = []
    n = len(pts)

    # Normal da reta de corte
    d_perp = np.array([-direction[1], direction[0]])

    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        # Distância com sinal de cada ponto à reta
        s1 = np.dot(p1 - origin, d_perp)
        s2 = np.dot(p2 - origin, d_perp)

        # Se os sinais diferem, a aresta cruza a reta
        if s1 * s2 < 0:
            # Interpolação linear para encontrar o ponto de cruzamento
            t = s1 / (s1 - s2)
            intersection = p1 + t * (p2 - p1)
            intersections.append(intersection)

    return intersections


def check_and_correct_inversion(mask: np.ndarray) -> np.ndarray:
    """
    Verifica se a máscara binária está invertida (objeto preto e fundo branco)
    e inverte se necessário.

    Como o objeto de interesse está posicionado no centro da imagem, as bordas
    exteriores da máscara devem corresponder ao fundo (cor preta / 0). Se a
    maioria dos pixels das bordas for branca (255), a máscara é invertida.
    """
    h, w = mask.shape[:2]
    # Definir uma borda de 5% da largura/altura
    border_y = max(1, int(h * 0.05))
    border_x = max(1, int(w * 0.05))

    # Extrair pixels da borda (topo, base, esquerda, direita)
    top_pixels = mask[0:border_y, :]
    bottom_pixels = mask[h-border_y:h, :]
    left_pixels = mask[:, 0:border_x]
    right_pixels = mask[:, w-border_x:w]

    # Calcular a média dos pixels de borda
    total_border_pixels = (top_pixels.size + bottom_pixels.size + 
                           left_pixels.size + right_pixels.size)
    white_border_pixels = (cv2.countNonZero(top_pixels) + 
                           cv2.countNonZero(bottom_pixels) + 
                           cv2.countNonZero(left_pixels) + 
                           cv2.countNonZero(right_pixels))

    border_white_ratio = white_border_pixels / total_border_pixels

    # Se mais de 50% dos pixels da borda são brancos, a máscara está invertida
    if border_white_ratio > 0.5:
        logger.info(
            f"  Detecção de inversão de máscara: {border_white_ratio:.1%} "
            f"dos pixels da borda são brancos. Invertendo máscara..."
        )
        return cv2.bitwise_not(mask)

    return mask


def refine_contour_with_edges(contour, gray, margin=30, mode="close") -> tuple:
    """
    Refina o contorno usando bordas Canny detectadas em uma ROI ao redor dele.
    Retorna o contorno refinado e o IoU (Intersection over Union).
    """
    bx, by, bw, bh = cv2.boundingRect(contour)
    h_img, w_img = gray.shape[:2]
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(w_img, bx + bw + margin)
    y2 = min(h_img, by + bh + margin)
    
    roi_gray = gray[y1:y2, x1:x2]
    
    # Threshold Otsu para o Canny
    high_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(roi_gray, low_thresh, high_thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if mode == "raw":
        edges_processed = edges
    elif mode == "close":
        edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    elif mode == "dilate_close":
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edges_processed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    else:
        edges_processed = edges
        
    contours_canny, _ = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_canny:
        return contour, 0.0
        
    contour_roi = contour.copy()
    contour_roi[:, :, 0] -= x1
    contour_roi[:, :, 1] -= y1
    orig_area = cv2.contourArea(contour_roi)
    
    best_iou = 0.0
    best_contour = None
    
    h_roi, w_roi = roi_gray.shape[:2]
    mask_orig = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.drawContours(mask_orig, [contour_roi], -1, 255, -1)
    
    for c_canny in contours_canny:
        # Filtrar contornos irrelevantes por área
        if cv2.contourArea(c_canny) < 0.3 * orig_area:
            continue
            
        mask_canny = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.drawContours(mask_canny, [c_canny], -1, 255, -1)
        
        intersection = cv2.bitwise_and(mask_orig, mask_canny)
        union = cv2.bitwise_or(mask_orig, mask_canny)
        
        area_intersection = np.sum(intersection > 0)
        area_union = np.sum(union > 0)
        
        if area_union > 0:
            iou = area_intersection / area_union
            if iou > best_iou:
                best_iou = iou
                best_contour = c_canny
                
    if best_contour is not None and best_iou > 0.5:
        refined_contour = best_contour.copy()
        refined_contour[:, :, 0] += x1
        refined_contour[:, :, 1] += y1
        return refined_contour, best_iou
    else:
        return contour, 0.0

def _preprocess_mdf_background(
    gray_blurred: np.ndarray,
    image_color: np.ndarray,
    calibration_corners: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta fundo de mesa branco ao redor da placa MDF e remove a borda queimada a laser.
    Também mascara o bloco de calibração se fornecido.
    Homogeniza as regiões externas/bloco com a cor mediana do MDF.
    """
    # Se o bloco de calibração foi detectado, removemos ele primeiro preenchendo com a cor mediana do MDF
    if calibration_corners is not None:
        try:
            x, y, cw, ch = cv2.boundingRect(calibration_corners.astype(np.int32))
            
            # Criar máscara para o bloco
            calib_mask = np.zeros(gray_blurred.shape, dtype=np.uint8)
            hull = cv2.convexHull(calibration_corners.astype(np.int32))
            cv2.drawContours(calib_mask, [hull], -1, 255, -1)
            
            # Dilatar a máscara para cobrir a borda de 3mm do bloco
            square_size_px = max(cw / 7.0, ch / 6.0)
            scale_px_mm = square_size_px / 6.0
            dilation_px = int(np.ceil(3.0 * scale_px_mm)) + 5  # 3mm de borda + 5px de margem
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
            calib_mask_dilated = cv2.dilate(calib_mask, kernel)
            
            # Copiar para modificação segura
            gray_out = gray_blurred.copy()
            color_out = image_color.copy()
            
            # Criar máscara temporária excluindo a região dilata e o fundo claro (>210)
            ex_mask = np.ones(gray_blurred.shape, dtype=np.uint8) * 255
            ex_mask[calib_mask_dilated == 255] = 0
            
            _, white_mask = cv2.threshold(gray_blurred, 210, 255, cv2.THRESH_BINARY)
            ex_mask[white_mask == 255] = 0
            
            # Calcular cor mediana do MDF livre de interferências
            mdf_gray_pixels = gray_blurred[ex_mask == 255]
            median_gray = int(np.median(mdf_gray_pixels)) if len(mdf_gray_pixels) > 0 else 149
            
            mdf_color_pixels = image_color[ex_mask == 255]
            if len(mdf_color_pixels) > 0:
                median_color = np.median(mdf_color_pixels, axis=0).astype(np.uint8)
            else:
                median_color = np.array([149, 149, 149], dtype=np.uint8)
                
            # Pintar a região do bloco de calibração com a cor mediana do MDF
            gray_out[calib_mask_dilated == 255] = median_gray
            color_out[calib_mask_dilated == 255] = median_color
            
            gray_blurred = gray_out
            image_color = color_out
            logger.info("  [Pre-processamento] Bloco de calibração mascarado com cor do MDF.")
        except Exception as e:
            logger.warning(f"  [Pre-processamento] Falha ao mascarar bloco de calibração: {e}")

    # Detectar o fundo branco da mesa
    _, thresh = cv2.threshold(gray_blurred, 210, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_blurred, image_color
        
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    total_area = gray_blurred.shape[0] * gray_blurred.shape[1]
    
    if (largest_area / total_area) > 0.30:
        logger.info(
            f"  [Pre-processamento] Fundo branco detectado ({largest_area/total_area*100:.1f}%). "
            f"Limpando bordas e homogenizando com cor do MDF..."
        )
        # Criar máscara para o MDF (255 = MDF, 0 = fora)
        mdf_mask = np.ones(gray_blurred.shape, dtype=np.uint8) * 255
        cv2.drawContours(mdf_mask, [largest_contour], -1, 0, -1)
        
        # Erosão de 1.5% da menor dimensão
        h, w = gray_blurred.shape[:2]
        min_dim = min(h, w)
        erosion_sz = int(np.round(min_dim * 0.015))
        if erosion_sz % 2 == 0:
            erosion_sz += 1
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_sz, erosion_sz))
        eroded_mdf_mask = cv2.erode(mdf_mask, kernel, iterations=1)
        
        # Cores medianas da região MDF
        mdf_gray_pixels = gray_blurred[eroded_mdf_mask == 255]
        median_gray = int(np.median(mdf_gray_pixels)) if len(mdf_gray_pixels) > 0 else 149
        
        mdf_color_pixels = image_color[eroded_mdf_mask == 255]
        if len(mdf_color_pixels) > 0:
            median_color = np.median(mdf_color_pixels, axis=0).astype(np.uint8)
        else:
            median_color = np.array([149, 149, 149], dtype=np.uint8)
            
        gray_out = gray_blurred.copy()
        color_out = image_color.copy()
        
        gray_out[eroded_mdf_mask == 0] = median_gray
        color_out[eroded_mdf_mask == 0] = median_color
        
        return gray_out, color_out
        
    return gray_blurred, image_color


# ──────────────────────────────────────────────────────────────────────────────
# Função principal de segmentação
# ──────────────────────────────────────────────────────────────────────────────

def segment(
    gray_blurred: np.ndarray,
    image_color: np.ndarray,
    image_name: str,
    background: np.ndarray = None,
    strategy: str = None,
    save_mask: bool = True,
    masks_dir: str = None,
    calibration_corners: np.ndarray = None
) -> list:
    """
    Segmenta o corpo de prova na imagem e extrai métricas geométricas.

    Args:
        gray_blurred: Imagem em escala de cinza pré-processada (uint8).
        image_color: Imagem colorida original (BGR, uint8) para estratégia LAB.
        image_name: Nome da imagem (usado para log e nome do arquivo da máscara).
        background: Imagem da base vazia (BGR, uint8). Necessário para
                    estratégia "background_sub". Default: None.
        strategy: Estratégia de segmentação. Default: config.SEGMENTATION_STRATEGY.
        save_mask: Se True, salva a máscara em data/masks/ para inspeção no ImageJ.
        masks_dir: Diretório para salvar máscaras. Default: config.MASKS_DIR.
        calibration_corners: Cantos detectados do bloco de calibração.

    Returns:
        Lista de dicts, um por contorno válido encontrado. Cada dict contém
        as métricas geométricas em pixels (ver extract_contour_metrics).

    Raises:
        SegmentationError: Se nenhum contorno válido for encontrado após
                           todas as estratégias. Inclui sugestões de ajuste.
    """
    strategy = strategy or config.SEGMENTATION_STRATEGY
    masks_dir = masks_dir or config.MASKS_DIR

    logger.info(f"  Segmentando '{image_name}' (estratégia: {strategy})")

    # Pré-processamento: isolar MDF e limpar o fundo / borda queimada a laser / bloco calib
    gray_blurred, image_color = _preprocess_mdf_background(gray_blurred, image_color, calibration_corners)

    mask = None

    if strategy == "auto":
        # Tentar cada estratégia na ordem de preferência
        strategies_to_try = []

        strategies_to_try.append(("yellow", None))
        strategies_to_try.append(("lab_b", None))
        if background is not None:
            strategies_to_try.append(("background_sub", background))
        strategies_to_try.append(("otsu_dark", None))
        strategies_to_try.append(("lab", None))
        strategies_to_try.append(("otsu", None))

        for strat_name, bg in strategies_to_try:
            logger.debug(f"  Auto: tentando '{strat_name}'...")
            try:
                candidate_mask = _apply_strategy(
                    strat_name, gray_blurred, image_color, bg
                )
                candidate_mask = check_and_correct_inversion(candidate_mask)
                candidate_mask = postprocess_mask(candidate_mask)
                quality = evaluate_mask_quality(candidate_mask)

                logger.debug(
                    f"  Auto '{strat_name}': proporção={quality['proportion']:.3f}, "
                    f"contornos={quality['n_contours']}, "
                    f"compacidade={quality['compactness']:.3f}, "
                    f"ok={quality['is_good']}"
                )

                if quality["is_good"]:
                    mask = candidate_mask
                    logger.info(f"  ✓ Estratégia selecionada: '{strat_name}'")
                    break
            except Exception as e:
                logger.debug(f"  Auto: '{strat_name}' falhou: {e}")
                continue

        if mask is None:
            _raise_segmentation_error(image_name, strategy, background)

    else:
        # Estratégia específica
        mask = _apply_strategy(strategy, gray_blurred, image_color, background)
        mask = check_and_correct_inversion(mask)
        mask = postprocess_mask(mask)

    # Detectar contornos
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtrar por área mínima, circularidade e aspect ratio (evita ruídos em forma de linha)
    valid_contours = []
    total_pixels = mask.shape[0] * mask.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_CONTOUR_AREA_PX:
            continue
            
        # Calcular aspect ratio do bounding box
        _, _, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 0
        
        # Calcular circularidade
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        if circularity < 0.05 or aspect_ratio > 5.0 or (area >= config.MAX_CONTOUR_AREA_PROPORTION * total_pixels and area >= 150000):
            logger.info(
                f"  Descartando contorno ruidoso/muito grande: área={area:.0f} px², "
                f"bbox={w}x{h} px, circularidade={circularity:.3f}, aspect_ratio={aspect_ratio:.2f}"
            )
            continue
            
        valid_contours.append(c)

    # Refinar contornos válidos usando as bordas Canny
    refined_contours = []
    for c in valid_contours:
        c_ref, iou = refine_contour_with_edges(c, gray_blurred, margin=30, mode="close")
        if iou > 0.5:
            _, _, w_orig, h_orig = cv2.boundingRect(c)
            _, _, w_ref, h_ref = cv2.boundingRect(c_ref)
            logger.info(
                f"    [Refinamento Canny] IoU={iou:.4f} -> BBox: {w_orig}x{h_orig} px -> {w_ref}x{h_ref} px"
            )
            refined_contours.append(c_ref)
        else:
            logger.info("    [Refinamento Canny] Falha ou IoU muito baixo. Mantendo original.")
            refined_contours.append(c)
    valid_contours = refined_contours

    # Ordenar contornos: priorizar o que tem melhor score (área × proximidade do centro)
    # Se houver apenas 1, será o primeiro de qualquer forma.
    if len(valid_contours) > 0:
        h, w = mask.shape[:2]
        cx_img, cy_img = w // 2, h // 2
        max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)
        
        def get_contour_score(c):
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                x, y, cw, ch = cv2.boundingRect(c)
                cx = x + cw / 2
                cy = y + ch / 2
                
            dist = np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
            center_score = 1.0 - (dist / max_dist)
            return area * center_score

        valid_contours = sorted(valid_contours, key=get_contour_score, reverse=True)
        if len(valid_contours) > 1:
            logger.info(
                f"  Ordenados {len(valid_contours)} contornos por score (área × centroide)."
            )

    if not valid_contours:
        _raise_segmentation_error(image_name, strategy, background)

    logger.info(
        f"  ✓ {len(valid_contours)} contorno(s) válido(s) detectado(s)"
    )

    # Salvar máscara para inspeção no ImageJ
    if save_mask:
        mask_dir = Path(masks_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{image_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        logger.debug(f"  Máscara salva: {mask_path}")

    # Extrair métricas de cada contorno
    results = []
    for i, contour in enumerate(valid_contours):
        metrics = extract_contour_metrics(contour)
        metrics["contour_index"] = i
        metrics["image_name"] = image_name
        results.append(metrics)

        logger.info(
            f"    Contorno {i}: área={metrics['area_px']:.0f} px², "
            f"bbox={metrics['bbox_w']}×{metrics['bbox_h']} px, "
            f"circularidade={metrics['circularity']:.3f}"
        )

    return results


def _apply_strategy(
    strategy: str,
    gray_blurred: np.ndarray,
    image_color: np.ndarray,
    background: np.ndarray = None
) -> np.ndarray:
    """Aplica uma estratégia de segmentação específica."""
    if strategy == "background_sub":
        if background is None:
            raise ValueError(
                "Estratégia 'background_sub' requer imagem de background. "
                "Forneça uma foto da base vazia."
            )
        return segment_background_sub(image_color, background)

    elif strategy == "lab":
        return segment_lab(image_color)

    elif strategy == "otsu":
        return segment_otsu(gray_blurred)

    elif strategy == "otsu_dark":
        return segment_otsu_dark(gray_blurred)

    elif strategy == "adaptive":
        return segment_adaptive(gray_blurred)

    elif strategy == "yellow":
        return segment_yellow(image_color)

    elif strategy == "lab_b":
        return segment_lab_b(image_color)

    else:
        raise ValueError(
            f"Estratégia de segmentação desconhecida: '{strategy}'. "
            f"Opções: background_sub, lab, otsu, adaptive, yellow, auto"
        )


def _raise_segmentation_error(
    image_name: str,
    strategy: str,
    background: np.ndarray = None
):
    """Lança SegmentationError com mensagem detalhada de diagnóstico."""
    suggestions = [
        f"Nenhum contorno válido encontrado em '{image_name}'.",
        "",
        "Sugestões de ajuste:",
    ]

    if background is not None:
        suggestions.extend([
            f"  - Ajuste BG_SUB_THRESHOLD no config.py (atual: {config.BG_SUB_THRESHOLD})",
            f"    ↓ Diminuir para recuperar bordas sutis da peça",
            f"    ↑ Aumentar para eliminar falsos positivos do fundo",
        ])
    else:
        suggestions.append(
            "  - Forneça uma foto da base vazia (background) para melhor segmentação"
        )

    suggestions.extend([
        f"  - Ajuste MIN_CONTOUR_AREA_PX (atual: {config.MIN_CONTOUR_AREA_PX})",
        f"    ↓ Diminuir se a peça é pequena na imagem",
        f"  - Ajuste MORPH_KERNEL_SIZE (atual: {config.MORPH_KERNEL_SIZE})",
        f"  - Tente estratégia diferente: --strategy lab|otsu|adaptive",
        f"  - Verifique a máscara em data/masks/ no ImageJ",
        f"  - Melhore a iluminação e o contraste da cena",
    ])

    raise SegmentationError("\n".join(suggestions))
