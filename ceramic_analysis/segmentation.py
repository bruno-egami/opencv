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



def segment_otsu(gray_blurred: np.ndarray, piece_is_lighter: bool = False) -> np.ndarray:
    """
    Segmentação por threshold automático de Otsu.

    Calcula o limiar ótimo que minimiza a variância intra-classe,
    assumindo uma distribuição bimodal (pico do fundo + pico do objeto).

    A polaridade (BINARY vs BINARY_INV) é determinada automaticamente:
    - piece_is_lighter=True  → peça clara (caulim) → THRESH_BINARY
    - piece_is_lighter=False → peça escura (marrom) → THRESH_BINARY_INV

    Args:
        gray_blurred: Imagem em escala de cinza com filtro gaussiano (uint8).
        piece_is_lighter: Se True, a peça é mais clara que o fundo.

    Returns:
        Máscara binária (uint8).
    """
    if piece_is_lighter:
        thresh_type = cv2.THRESH_BINARY
        polarity_label = "BINARY (peça clara)"
    else:
        thresh_type = cv2.THRESH_BINARY_INV
        polarity_label = "BINARY_INV (peça escura)"

    if config.MANUAL_THRESHOLD is not None:
        thresh_val = config.MANUAL_THRESHOLD
        
        # Blur forte local para evitar bordas serrilhadas e ruído do MDF
        heavy_blur = cv2.GaussianBlur(gray_blurred, (31, 31), 0)
        _, mask = cv2.threshold(heavy_blur, thresh_val, 255, thresh_type)
        logger.debug(f"  Threshold manual: {thresh_val} ({polarity_label})")
    else:
        # Blur forte local para evitar bordas serrilhadas e ruído do MDF
        heavy_blur = cv2.GaussianBlur(gray_blurred, (31, 31), 0)
        thresh_val, mask = cv2.threshold(heavy_blur, 0, 255,
                                         thresh_type + cv2.THRESH_OTSU)
        logger.debug(f"  Threshold Otsu automático: {thresh_val} ({polarity_label})")

    return mask


def segment_otsu_dark(gray_blurred: np.ndarray, piece_is_lighter: bool = False) -> np.ndarray:
    """
    Segmentação por Otsu considerando apenas pixels escuros (< 160).
    Exclui pixels muito claros da calibração (bloco branco/grade) para
    evitar que enviesem o limiar de Otsu.
    
    A polaridade é determinada pelo parâmetro piece_is_lighter.
    """
    if piece_is_lighter:
        thresh_type = cv2.THRESH_BINARY
        polarity_label = "BINARY (peça clara)"
    else:
        thresh_type = cv2.THRESH_BINARY_INV
        polarity_label = "BINARY_INV (peça escura)"

    # Blur forte local para evitar bordas serrilhadas e ruído do MDF
    heavy_blur = cv2.GaussianBlur(gray_blurred, (31, 31), 0)

    # Selecionar pixels < 160
    pixels = heavy_blur[heavy_blur < 160]
    
    if len(pixels) == 0:
        # Fallback para Otsu padrão se não houver pixels escuros
        _, mask = cv2.threshold(heavy_blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
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
    
    for t in range(160):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
            
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        
        var_between = w_b * w_f * ((m_b - m_f) ** 2)
        
        if var_between > current_max:
            current_max = var_between
            thresh_val = t
            
    logger.debug(f"  Otsu Dark: limiar calculado = {thresh_val} ({polarity_label})")
    
    _, mask = cv2.threshold(heavy_blur, thresh_val, 255, thresh_type)
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


def segment_grabcut_seeded(
    image_color: np.ndarray,
    seed_point: tuple,
    mdf_point: tuple,
    iterations: int = 5,
    max_dim: int = 1000,
    calibration_corners: np.ndarray = None
) -> np.ndarray:
    """
    Segmentação via GrabCut inicializado com pontos-semente do usuário.
    
    Para performance, a imagem é redimensionada para max_dim pixels na maior
    dimensão antes do GrabCut, e a máscara resultante é escalada de volta.
    
    Inicialização da máscara:
    1. Tudo como GC_PR_BGD (provável background)
    2. Borda de 10% como GC_BGD (background definitivo)
    3. Região ampla ao redor do seed_point como GC_PR_FGD
    4. Núcleo do seed_point como GC_FGD (foreground definitivo)
    5. Região do mdf_point como GC_BGD
    
    Args:
        image_color: Imagem BGR (uint8).
        seed_point: (x, y) ponto dentro da peça (foreground).
        mdf_point: (x, y) ponto no MDF (background).
        iterations: Número de iterações do GrabCut.
        max_dim: Dimensão máxima da imagem redimensionada para GrabCut.
        
    Returns:
        Máscara binária (uint8): 255 = objeto, 0 = fundo.
    """
    h_orig, w_orig = image_color.shape[:2]
    
    # Calcular fator de escala para redimensionamento
    scale = min(max_dim / max(h_orig, w_orig), 1.0)
    
    if scale < 1.0:
        w_small = int(w_orig * scale)
        h_small = int(h_orig * scale)
        img_small = cv2.resize(image_color, (w_small, h_small), interpolation=cv2.INTER_AREA)
        # Escalar seed points
        sp = (int(seed_point[0] * scale), int(seed_point[1] * scale))
        mp = (int(mdf_point[0] * scale), int(mdf_point[1] * scale))
        logger.debug(f"  GrabCut: redimensionando {w_orig}x{h_orig} → {w_small}x{h_small} (scale={scale:.3f})")
    else:
        img_small = image_color
        w_small, h_small = w_orig, h_orig
        sp = seed_point
        mp = mdf_point
    
    # Inicializar máscara: tudo como provável background
    gc_mask = np.full((h_small, w_small), cv2.GC_PR_BGD, dtype=np.uint8)
    
    # Marcar borda de 10% como background definitivo (a peça não está na borda)
    border_y = max(5, int(h_small * 0.10))
    border_x = max(5, int(w_small * 0.10))
    gc_mask[:border_y, :] = cv2.GC_BGD          # Topo
    gc_mask[h_small - border_y:, :] = cv2.GC_BGD  # Base
    gc_mask[:, :border_x] = cv2.GC_BGD            # Esquerda
    gc_mask[:, w_small - border_x:] = cv2.GC_BGD  # Direita
    
    # Raio para as sementes — proporcional à imagem redimensionada
    seed_radius = max(8, int(min(h_small, w_small) * 0.02))
    
    # Marcar região ampla ao redor do seed como provável foreground
    fgd_radius = max(seed_radius * 5, int(min(h_small, w_small) * 0.12))
    cv2.circle(gc_mask, sp, fgd_radius, cv2.GC_PR_FGD, -1)
    
    # Marcar núcleo do seed_point como foreground definitivo
    cv2.circle(gc_mask, sp, seed_radius, cv2.GC_FGD, -1)
    
    # Marcar região do mdf_point como background definitivo
    cv2.circle(gc_mask, mp, seed_radius * 2, cv2.GC_BGD, -1)
    
    # Marcar região do bloco de calibração como background definitivo
    if calibration_corners is not None:
        try:
            hull = cv2.convexHull(calibration_corners.astype(np.int32))
            if scale < 1.0:
                hull = (hull * scale).astype(np.int32)
            
            calib_mask = np.zeros_like(gc_mask)
            cv2.drawContours(calib_mask, [hull], -1, 255, -1)
            
            # Dilatar para cobrir a borda branca (aprox 3mm)
            x, y, w, h = cv2.boundingRect(hull)
            square_size = max(w / 7.0, h / 6.0)
            dilation = int(np.ceil(3.0 * (square_size / 6.0))) + int(10 * scale)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
            calib_mask_dilated = cv2.dilate(calib_mask, kernel)
            
            gc_mask[calib_mask_dilated == 255] = cv2.GC_BGD
            logger.debug("  GrabCut: Bloco de calibração marcado como GC_BGD.")
        except Exception as e:
            logger.warning(f"  GrabCut: Erro ao mascarar bloco calibração: {e}")
    
    # Modelos GMM para foreground e background
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    # Executar GrabCut
    logger.debug(f"  GrabCut: seed={sp}, mdf={mp}, iter={iterations}, img={w_small}x{h_small}")
    cv2.grabCut(
        img_small, gc_mask, None,
        bgd_model, fgd_model,
        iterations,
        cv2.GC_INIT_WITH_MASK
    )
    
    # Converter resultado: FGD e PR_FGD → 255, resto → 0
    mask_small = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    
    # Escalar máscara de volta para resolução original
    if scale < 1.0:
        mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small
    
    return mask

def segment_edges(gray_blurred: np.ndarray, seed_point: tuple = None, calibration_corners: np.ndarray = None) -> np.ndarray:
    """
    Segmentação robusta usando Canny Edges e validada pelo clique do usuário.
    Projetado especificamente para argilas/caulim que soltam pó na base MDF.
    """
    h_orig, w_orig = gray_blurred.shape[:2]
    scale = 800.0 / max(h_orig, w_orig)
    if scale < 1.0:
        gray_small = cv2.resize(gray_blurred, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray_blurred.copy()

    # Suaviza a imagem levemente para reduzir ruído e textura (pó de argila),
    # mas sem encolher as sombras verdadeiras da borda da peça (grau 9 em vez de 15)
    blur = cv2.GaussianBlur(gray_small, (9, 9), 0)
    
    # Extrair bordas físicas (sombra e quina da peça)
    edges = cv2.Canny(blur, 30, 100)

    # Fechamento morfológico forte para CONECTAR as linhas e garantir que buracos na sombra se fechem
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Ocultar o bloco de calibração se estiver disponível para evitar falsos positivos
    if calibration_corners is not None and len(calibration_corners) > 0:
        try:
            hull = cv2.convexHull(calibration_corners)
            if scale < 1.0:
                hull = (hull * scale).astype(np.int32)
            
            x, y, w, h = cv2.boundingRect(hull)
            square_size = max(w / 7.0, h / 6.0)
            
            # Dilatação massiva para ter CERTEZA que engolirá as "bordas artificiais" do pré-processamento
            dilation = int(np.ceil(3.0 * (square_size / 6.0))) + int(10 * scale) + 50
            
            # Pintar um retângulo totalmente preto sobre o bloco
            cv2.rectangle(closed_edges, (int(x)-dilation, int(y)-dilation), (int(x+w)+dilation, int(y+h)+dilation), 0, -1)
        except Exception as e:
            logger.warning(f"  Edges: Erro ao mascarar bloco de calibração: {e}")

    # Achar todos os contornos da imagem (usar RETR_TREE para pegar contornos internos e externos)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_small = np.zeros_like(gray_small)
    
    # Se o seed point estiver disponível, achamos o MENOR contorno que abriga o ponto
    if seed_point is not None and len(contours) > 0:
        sp_small = (seed_point[0] * scale, seed_point[1] * scale)
        
        valid_contours = []
        img_area = gray_small.shape[0] * gray_small.shape[1]
        for c in contours:
            # pointPolygonTest retorna >= 0 se o ponto está dentro ou na borda do contorno
            if cv2.pointPolygonTest(c, sp_small, False) >= 0:
                area = cv2.contourArea(c)
                # Adicionamos uma área mínima para não pegar ruídos de 1 pixel
                # E área máxima (15% da imagem) para excluir a placa de MDF inteira
                if 100 < area < (img_area * 0.15):
                    valid_contours.append((area, c))
                
        if valid_contours:
            # Ordena de forma DECRESCENTE para pegar o MAIOR contorno do objeto.
            # Isso garante que a linha capturada seja a aresta externa (base da peça no MDF)
            # e não a aresta interna (teto da peça volumétrica).
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            best_contour = valid_contours[0][1]
            cv2.drawContours(mask_small, [best_contour], -1, 255, -1)
            logger.debug(f"  Edges: Contorno externo validado pelo clique (área={valid_contours[0][0]:.1f}).")
        else:
            # Se não achou, recai sobre o maior contorno da imagem inteira como fallback
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_small, [c], -1, 255, -1)
            logger.debug("  Edges: Seed point fora de qualquer contorno válido. Fallback para maior contorno.")
    elif len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_small, [c], -1, 255, -1)

    # Escalar a máscara de volta para o tamanho original
    if scale < 1.0:
        mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small

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
    # Imagens de alta resolução (~64MP) fazem com que a peça de argila caia para < 0.5%
    proportion_ok = 0.0001 <= proportion <= 0.80
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
    Aplica fechamento morfológico para eliminar lacunas nas bordas, e
    abertura morfológica para raspar pequenas oscilações/rebarbas físicas.

    Isso fecha pequenas lacunas causadas por reflexos, e remove
    pequenos inchaços na borda para que a Bounding Box cole perfeitamente
    na linha reta principal da peça.
    """
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE
    )
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close,
        iterations=config.MORPH_ITERATIONS
    )
    
    # Burr Shaver opcional
    if getattr(config, "BURR_SHAVER_ENABLED", False):
        size = getattr(config, "BURR_SHAVER_SIZE", 201)
        # O kernel deve ter dimensão ímpar
        if size % 2 == 0:
            size += 1
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        return opened
        
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
    
    # Dimensões Robustas (Mediana) e deslocamento de centro
    robust_w, robust_h, robust_center = compute_robust_dimensions(contour, min_rect)

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
        # Dimensões Robustas
        "robust_w": robust_w,
        "robust_h": robust_h,
        "robust_center_x": robust_center[0],
        "robust_center_y": robust_center[1],
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


def compute_robust_dimensions(contour: np.ndarray, min_rect: tuple) -> tuple:
    """
    Calcula as dimensões e o centro robusto (Mediana) da borda da peça,
    ignorando rebarbas ou cantos arredondados, pegando a mediana da linha de contorno.
    """
    center, (rect_w, rect_h), angle = min_rect
    
    # Rotacionar os pontos do contorno para ficarem alinhados aos eixos X e Y
    pts = contour.reshape(-1, 2) - np.array(center)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    
    # A matriz R gira os pontos do contorno para o eixo upright
    R = np.array(((c, -s), (s, c)))
    pts_rot = np.dot(pts, R)
    
    # Encontrar as extremidades absolutas do contorno rotacionado
    min_x, max_x = np.min(pts_rot[:, 0]), np.max(pts_rot[:, 0])
    min_y, max_y = np.min(pts_rot[:, 1]), np.max(pts_rot[:, 1])
    
    # Isolar os pontos que pertencem a cada uma das 4 faces
    # Ignoramos os 25% das extremidades (cantos) para pegar apenas a face plana central!
    margin_w = rect_w * 0.25
    margin_h = rect_h * 0.25
    
    # Para a borda superior, pegamos os pontos com Y baixo, e X no meio da peça
    top_pts = pts_rot[(pts_rot[:, 1] < min_y + rect_h * 0.15) & 
                      (pts_rot[:, 0] > min_x + margin_w) & 
                      (pts_rot[:, 0] < max_x - margin_w)]
                      
    bottom_pts = pts_rot[(pts_rot[:, 1] > max_y - rect_h * 0.15) & 
                         (pts_rot[:, 0] > min_x + margin_w) & 
                         (pts_rot[:, 0] < max_x - margin_w)]
                         
    left_pts = pts_rot[(pts_rot[:, 0] < min_x + rect_w * 0.15) & 
                       (pts_rot[:, 1] > min_y + margin_h) & 
                       (pts_rot[:, 1] < max_y - margin_h)]
                       
    right_pts = pts_rot[(pts_rot[:, 0] > max_x - rect_w * 0.15) & 
                        (pts_rot[:, 1] > min_y + margin_h) & 
                        (pts_rot[:, 1] < max_y - margin_h)]
    
    # Se a máscara falhar, retorna o retângulo original
    if len(top_pts) == 0 or len(bottom_pts) == 0 or len(left_pts) == 0 or len(right_pts) == 0:
        return float(rect_w), float(rect_h), center
        
    # A posição real (mediana) de cada face
    med_top = np.median(top_pts[:, 1])
    med_bot = np.median(bottom_pts[:, 1])
    med_left = np.median(left_pts[:, 0])
    med_right = np.median(right_pts[:, 0])
    
    robust_w = med_right - med_left
    robust_h = med_bot - med_top
    
    # O centro do retângulo robusto em relação à origem local
    robust_cx = (med_left + med_right) / 2.0
    robust_cy = (med_top + med_bot) / 2.0
    
    # Girar o deslocamento do centro de volta para o sistema de coordenadas original
    shift = np.array([robust_cx, robust_cy])
    R_inv = np.array(((c, s), (-s, c)))
    shift_orig = np.dot(shift, R_inv)
    
    robust_center = (center[0] + shift_orig[0], center[1] + shift_orig[1])
    
    # Garantir orientação correta
    if (rect_w > rect_h and robust_w < robust_h) or (rect_w <= rect_h and robust_w >= robust_h):
        robust_w, robust_h = robust_h, robust_w
        
    return float(robust_w), float(robust_h), robust_center

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
        positions: Lista de posições relativas (0.0 a 1.0). Default: [0.20, 0.50, 0.80].

    Returns:
        Dict com as medidas em pixels:
            - cross_width_20pct_px, cross_width_50pct_px, cross_width_80pct_px
            - cross_length_20pct_px, cross_length_50pct_px, cross_length_80pct_px
            - cross_width_pts (lista de pares de pontos para desenho)
            - cross_length_pts (lista de pares de pontos para desenho)
        Retorna None se o contorno for insuficiente.
    """
    if positions is None:
        positions = [0.20, 0.50, 0.80]

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
    calibration_corners: np.ndarray = None,
    seed_point: tuple = None
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
    
    # Proteger região ao redor do seed_point (evita mascarar peças claras como caulim)
    if seed_point is not None:
        protection_radius = max(50, int(min(gray_blurred.shape[:2]) * 0.15))
        cv2.circle(thresh, seed_point, protection_radius, 0, -1)
    
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
    calibration_corners: np.ndarray = None,
    seed_point: tuple = None,
    mdf_point: tuple = None
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
    gray_blurred, image_color = _preprocess_mdf_background(gray_blurred, image_color, calibration_corners, seed_point)

    # Determinar se a peça é mais clara que o fundo usando os seed points
    piece_is_lighter = False
    if seed_point is not None and mdf_point is not None:
        gray_for_brightness = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) if len(image_color.shape) == 3 else gray_blurred
        # Amostrar patch 31x31 ao redor de cada ponto
        patch_r = 15
        h_img, w_img = gray_for_brightness.shape[:2]
        
        px, py = seed_point
        y1p = max(0, py - patch_r)
        y2p = min(h_img, py + patch_r + 1)
        x1p = max(0, px - patch_r)
        x2p = min(w_img, px + patch_r + 1)
        piece_brightness = float(np.mean(gray_for_brightness[y1p:y2p, x1p:x2p]))
        
        mx, my = mdf_point
        y1m = max(0, my - patch_r)
        y2m = min(h_img, my + patch_r + 1)
        x1m = max(0, mx - patch_r)
        x2m = min(w_img, mx + patch_r + 1)
        mdf_brightness = float(np.mean(gray_for_brightness[y1m:y2m, x1m:x2m]))
        
        piece_is_lighter = piece_brightness > mdf_brightness
        logger.info(
            f"  [Seed] Luminância peça={piece_brightness:.1f}, MDF={mdf_brightness:.1f} "
            f"→ peça {'MAIS CLARA' if piece_is_lighter else 'MAIS ESCURA'} que o fundo"
        )

    mask = None

    if strategy == "auto":
        # Tentar cada estratégia na ordem de preferência
        strategies_to_try = []

        if getattr(config, "MATERIAL_TYPE", "Argila") == "Argila":
            strategies_to_try.append(("edges", None))

        if seed_point is not None and mdf_point is not None:
            # Watershed resolve maravilhosamente bem o problema de peças com sombras
            # e núcleos com cor idêntica ao fundo, baseando-se em gradientes.
            strategies_to_try.append(("watershed_seeded", None))

        if background is not None:
            strategies_to_try.append(("background_sub", background))
        
        # Agora que temos piece_is_lighter determinado por sementes dinâmicas,
        # otsu_dark / otsu se tornam extremamente robustos para qualquer cor (clara ou escura)
        strategies_to_try.append(("otsu_dark", None))
        strategies_to_try.append(("lab", None))
        strategies_to_try.append(("otsu", None))

        for strat_name, bg in strategies_to_try:
            logger.debug(f"  Auto: tentando '{strat_name}'...")
            try:
                candidate_mask = _apply_strategy(
                    strat_name, gray_blurred, image_color, bg,
                    piece_is_lighter=piece_is_lighter,
                    seed_point=seed_point,
                    mdf_point=mdf_point,
                    calibration_corners=calibration_corners
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
        mask = _apply_strategy(
            strategy, gray_blurred, image_color, background,
            piece_is_lighter=piece_is_lighter,
            seed_point=seed_point,
            mdf_point=mdf_point,
            calibration_corners=calibration_corners
        )
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

    # Filtrar contornos pelo seed_point: se disponível, priorizar o que contém o seed
    if seed_point is not None and len(valid_contours) > 1:
        containing = []
        not_containing = []
        for c in valid_contours:
            if cv2.pointPolygonTest(c, (float(seed_point[0]), float(seed_point[1])), False) >= 0:
                containing.append(c)
            else:
                not_containing.append(c)
        if containing:
            logger.info(f"  [Seed] {len(containing)} contorno(s) contém o ponto-semente. Priorizando.")
            valid_contours = containing + not_containing

    # Ordenar contornos: priorizar o que tem melhor score (área × proximidade do centro)
    if len(valid_contours) > 0:
        h, w = mask.shape[:2]
        cx_img, cy_img = w // 2, h // 2
        max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)
        
        # Se temos seed_point, usar ele como referência de centralidade
        ref_x = float(seed_point[0]) if seed_point is not None else cx_img
        ref_y = float(seed_point[1]) if seed_point is not None else cy_img
        
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
                
            dist = np.sqrt((cx - ref_x) ** 2 + (cy - ref_y) ** 2)
            center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
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


def segment_watershed_seeded(
    image_color: np.ndarray,
    seed_point: tuple,
    mdf_point: tuple,
    max_dim: int = 1000,
    calibration_corners: np.ndarray = None
) -> np.ndarray:
    """
    Segmentação via Watershed inicializado por sementes.
    
    Excelente para resolver problemas de sombras projetadas: enquanto Otsu ou 
    GrabCut agrupam cores parecidas (como a sombra e a peça), o Watershed 
    flui a partir da semente e para EXATAMENTE na crista do gradiente 
    (a borda física da peça).
    """
    h_orig, w_orig = image_color.shape[:2]
    
    # Downsampling para velocidade e suavização de ruído fino
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
    
    if scale < 1.0:
        img_small = cv2.resize(image_color, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = image_color.copy()
        
    h_small, w_small = img_small.shape[:2]
    
    # Bilateral Filter: suaviza a granulação do MDF sem destruir (borrar) as bordas físicas!
    # Isso impede que o gradiente da peça se expanda para a sombra.
    img_small = cv2.bilateralFilter(img_small, 9, 75, 75)
    
    # Sementes redimensionadas
    sp = (int(seed_point[0] * scale), int(seed_point[1] * scale))
    mp = (int(mdf_point[0] * scale), int(mdf_point[1] * scale))
    seed_radius = max(5, int(min(h_small, w_small) * 0.015))
    
    # Marcadores para o Watershed:
    # 0 = Desconhecido (onde o algoritmo vai decidir)
    # 1 = Foreground (Peça)
    # 2 = Background (MDF)
    markers = np.zeros((h_small, w_small), dtype=np.int32)
    
    # Marcar foreground (1) no ponto da peça
    cv2.circle(markers, sp, seed_radius, 1, -1)
    
    # Marcar background (2) no ponto do MDF
    cv2.circle(markers, mp, seed_radius * 2, 2, -1)
    
    # Marcar bordas da imagem como background garantido (2)
    border_w = max(5, int(w_small * 0.02))
    border_h = max(5, int(h_small * 0.02))
    markers[:border_h, :] = 2
    markers[-border_h:, :] = 2
    markers[:, :border_w] = 2
    markers[:, -border_w:] = 2
    
    # Marcar bloco de calibração como background garantido (2)
    if calibration_corners is not None:
        try:
            hull = cv2.convexHull(calibration_corners.astype(np.int32))
            if scale < 1.0:
                hull = (hull * scale).astype(np.int32)
            # Preencher o casco convexo como background
            cv2.drawContours(markers, [hull], -1, 2, -1)
        except Exception as e:
            logger.warning(f"  Watershed: Erro ao marcar calibração: {e}")
            
    # Executar Watershed (modifica a matriz 'markers' inplace)
    cv2.watershed(img_small, markers)
    
    # Criar máscara binária apenas da área classificada como 1 (Foreground)
    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)
    mask_small[markers == 1] = 255
    
    # Upsampling da máscara
    if scale < 1.0:
        # Usa INTER_LINEAR para criar bordas suaves (anti-aliasing) em vez de blocos serrilhados (INTER_NEAREST)
        mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        # Aplica um pequeno desfoque para suavizar completamente as escadas de upscaling
        blur_size = max(3, int(1 / scale) * 2 + 1) # ~13 para scale=1/6
        if blur_size % 2 == 0: blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        # Binariza novamente para ter bordas precisas e perfeitamente lisas
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = mask_small
        
    return mask

def _apply_strategy(
    strategy: str,
    gray_blurred: np.ndarray,
    image_color: np.ndarray,
    background: np.ndarray = None,
    piece_is_lighter: bool = False,
    seed_point: tuple = None,
    mdf_point: tuple = None,
    calibration_corners: np.ndarray = None
) -> np.ndarray:
    """Aplica uma estratégia de segmentação específica."""
    if strategy == "watershed_seeded":
        if seed_point is None or mdf_point is None:
            raise ValueError("Estratégia 'watershed_seeded' requer seed_point e mdf_point.")
        return segment_watershed_seeded(
            image_color, seed_point, mdf_point,
            calibration_corners=calibration_corners
        )

    elif strategy == "grabcut_seeded":
        if seed_point is None or mdf_point is None:
            raise ValueError("Estratégia 'grabcut_seeded' requer seed_point e mdf_point.")
        return segment_grabcut_seeded(
            image_color, seed_point, mdf_point,
            iterations=5, max_dim=1000,
            calibration_corners=calibration_corners
        )

    elif strategy == "edges":
        return segment_edges(gray_blurred, seed_point, calibration_corners)

    elif strategy == "background_sub":
        if background is None:
            raise ValueError(
                "Estratégia 'background_sub' requer imagem de background. "
                "Forneça uma foto da base vazia."
            )
        return segment_background_sub(image_color, background)

    elif strategy == "lab":
        return segment_lab(image_color)

    elif strategy == "otsu":
        return segment_otsu(gray_blurred, piece_is_lighter=piece_is_lighter)

    elif strategy == "otsu_dark":
        return segment_otsu_dark(gray_blurred, piece_is_lighter=piece_is_lighter)

    elif strategy == "adaptive":
        return segment_adaptive(gray_blurred)

    elif strategy == "yellow":
        return segment_yellow(image_color)

    elif strategy == "lab_b":
        return segment_lab_b(image_color)

    else:
        raise ValueError(
            f"Estratégia de segmentação desconhecida: '{strategy}'. "
            f"Opções: grabcut_seeded, background_sub, lab, otsu, adaptive, yellow, auto"
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
