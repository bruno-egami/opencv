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


def fill_hollow_mask(mask: np.ndarray) -> np.ndarray:
    """
    Preenche o interior de peças ocas ou impressas em modo vaso.
    Aplica fechamento morfológico para conectar eventuais fendas nas paredes
    e preenche todos os contornos internos (furos).
    """
    filled = mask.copy()
    # 1. Fechamento morfológico forte para unir paredes desconexas (gap < ~15mm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    
    # 2. Preencher buracos internos
    contours_fill, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours_fill is not None and len(contours_fill) > 0:
        cv2.drawContours(filled, contours_fill, -1, 255, -1)

    # 3. Aplicar Convex Hull para garantir que o contorno em modo vaso
    # seja fechado em uma geometria sólida mesmo se apenas fragmentos do anel foram identificados
    contours_ext, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_ext) > 0:
        all_points = np.vstack(contours_ext)
        hull = cv2.convexHull(all_points)
        hull_mask = np.zeros_like(filled)
        cv2.drawContours(hull_mask, [hull], 0, 255, -1)
        return hull_mask
        
    return filled


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
        thresh_val, _ = cv2.threshold(heavy_blur, 0, 255,
                                         thresh_type + cv2.THRESH_OTSU)
                                         
        # Viés para ignorar reflexos sutis (como borda da fita azul ou iluminação na mesa)
        # O Otsu acha o vale perfeito, mas se o fundo tem reflexos, eles caem logo acima do vale.
        # Empurrar o threshold levemente em direção à peça elimina esses ruídos físicos.
        bias = 15
        if piece_is_lighter:
            thresh_val_biased = min(255, thresh_val + bias)
        else:
            thresh_val_biased = max(0, thresh_val - bias)
            
        _, mask = cv2.threshold(heavy_blur, thresh_val_biased, 255, thresh_type)
        
        logger.debug(f"  Threshold Otsu automático: {thresh_val} -> ajustado para {thresh_val_biased} ({polarity_label})")

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
    seed_points: list,
    mdf_points: list,
    iterations: int = 5,
    max_dim: int = 1000,
    calibration_corners: np.ndarray = None
) -> np.ndarray:
    """
    Segmentação via GrabCut inicializado com múltiplos pontos-semente do usuário.
    
    Para performance, a imagem é redimensionada para max_dim pixels na maior
    dimensão antes do GrabCut, e a máscara resultante é escalada de volta.
    
    Inicialização da máscara:
    1. Tudo como GC_PR_BGD (provável background)
    2. Borda de 5% como GC_BGD (background definitivo)
    3. Para cada semente:
       - Região ampla ao redor como GC_PR_FGD
       - Núcleo como GC_FGD (foreground definitivo)
    4. Região do mdf_points como GC_BGD
    
    Args:
        image_color: Imagem BGR (uint8).
        seed_points: Lista de tuplas (x, y) pontos dentro das peças (foreground).
        mdf_points: (x, y) ponto no MDF (background).
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
        sps = [(int(sp[0] * scale), int(sp[1] * scale)) for sp in seed_points]
        mps = [(int(p[0] * scale), int(p[1] * scale)) for p in mdf_points]
        logger.debug(f"  GrabCut: redimensionando {w_orig}x{h_orig} → {w_small}x{h_small} (scale={scale:.3f})")
    else:
        img_small = image_color.copy()
        w_small, h_small = w_orig, h_orig
        sps = seed_points
        mps = mdf_points
    
    # Inicializar máscara: tudo como provável background
    gc_mask = np.full((h_small, w_small), cv2.GC_PR_BGD, dtype=np.uint8)
    
    # Marcar borda de 5% como background definitivo
    border_y = max(5, int(h_small * 0.05))
    border_x = max(5, int(w_small * 0.05))
    gc_mask[:border_y, :] = cv2.GC_BGD          # Topo
    gc_mask[h_small - border_y:, :] = cv2.GC_BGD  # Base
    gc_mask[:, :border_x] = cv2.GC_BGD            # Esquerda
    gc_mask[:, w_small - border_x:] = cv2.GC_BGD  # Direita
    
    # Raio para as sementes (reduzido de ~10px para 3px para evitar vazamentos nas bordas)
    seed_radius = 3
    fgd_radius = max(seed_radius * 5, int(min(h_small, w_small) * 0.12))
    
    for sp in sps:
        # Marcar região ampla ao redor do seed como provável foreground
        cv2.circle(gc_mask, sp, fgd_radius, cv2.GC_PR_FGD, -1)
        # Marcar núcleo do seed_point como foreground definitivo
        cv2.circle(gc_mask, sp, seed_radius, cv2.GC_FGD, -1)
    
    # Marcar região do mdf_points como background definitivo
    for mp in mps:
        cv2.circle(gc_mask, mp, seed_radius * 3, cv2.GC_BGD, -1)
    
    # Marcar região do bloco de calibração como background definitivo
    if calibration_corners is not None:
        try:
            hull = cv2.convexHull(calibration_corners.astype(np.int32))
            if scale < 1.0:
                hull = (hull * scale).astype(np.int32)
            
            calib_mask = np.zeros_like(gc_mask)
            cv2.drawContours(calib_mask, [hull], -1, 255, -1)
            
            # Dilatar para cobrir margem
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            calib_mask = cv2.dilate(calib_mask, kernel)
            
            gc_mask[calib_mask == 255] = cv2.GC_BGD
        except Exception as e:
            logger.warning(f"  GrabCut: Erro ao marcar calibração: {e}")
            
    # Modelos de cor internos do GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Executar GrabCut no modo de máscara
    try:
        cv2.grabCut(img_small, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        logger.error(f"  Falha no GrabCut: {e}")
        raise ValueError("Falha na execução do GrabCut.")
        
    # Converter máscara GrabCut (0,2 = Fundo | 1,3 = Objeto) para Binária (0, 255)
    mask_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Aplicar fechamento morfológico para unir partes desconexas das peças
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, close_kernel)
    
    # Upsampling
    if scale < 1.0:
        mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        blur_size = max(3, int(1 / scale) * 2 + 1)
        if blur_size % 2 == 0: blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = mask_small
        
    return mask

def segment_edges(gray_blurred: np.ndarray, seed_points: list = None, calibration_corners: np.ndarray = None) -> np.ndarray:
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
    
    # Se os seed points estiverem disponíveis, achamos os contornos que os abrigam
    if seed_points is not None and len(contours) > 0:
        valid_contours = []
        img_area = gray_small.shape[0] * gray_small.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            # Adicionamos uma área mínima para não pegar ruídos de 1 pixel
            # E área máxima (25% da imagem) para excluir a placa de MDF inteira
            if 100 < area < (img_area * 0.25):
                # Verifica se ESSE contorno abriga ALGUMA das sementes
                contains_seed = False
                for sp in seed_points:
                    sp_small = (sp[0] * scale, sp[1] * scale)
                    if cv2.pointPolygonTest(c, sp_small, False) >= 0:
                        contains_seed = True
                        break
                
                if contains_seed:
                    valid_contours.append((area, c))
                
        if valid_contours:
            # Ordena de forma DECRESCENTE para pegar os maiores contornos
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            for area, best_contour in valid_contours:
                cv2.drawContours(mask_small, [best_contour], -1, 255, -1)
            logger.debug(f"  Edges: {len(valid_contours)} contorno(s) validados pelo(s) clique(s).")
        else:
            # Se não achou, recai sobre o maior contorno da imagem inteira como fallback
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_small, [c], -1, 255, -1)
            logger.debug("  Edges: Seed points fora de qualquer contorno válido. Fallback para maior contorno.")
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

    min_area = getattr(config, 'MIN_CONTOUR_AREA_PX', 5000)
    if getattr(config, 'HOLLOW_SPECIMEN', False):
        min_area = 500

    # Filtrar por área mínima, circularidade e aspect_ratio
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            _, _, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity >= 0.05 and aspect_ratio <= 10.0:
                # Rejeitar contornos que cobrem a maior parte de ambas as dimensoes da imagem (MDF de fundo)
                # ou que excedam o limite de área máxima
                max_prop = 0.95 if getattr(config, 'HOLLOW_SPECIMEN', False) else config.MAX_CONTOUR_AREA_PROPORTION
                
                # Tentar carregar resolução da calibração para normalizar os limites de área
                full_image_area = total_pixels
                try:
                    import os
                    if os.path.exists(config.CALIBRATION_FILE):
                        fs = cv2.FileStorage(config.CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
                        w_node = fs.getNode("image_width")
                        h_node = fs.getNode("image_height")
                        if not w_node.empty() and not h_node.empty():
                            full_image_area = int(w_node.real()) * int(h_node.real())
                        fs.release()
                except Exception:
                    pass

                resolution_factor = max(1.0, full_image_area / 10000000.0)
                max_area_const = 150000 * resolution_factor

                if (w < 0.8 * mask.shape[1] or h < 0.8 * mask.shape[0]) and (area < max_prop * full_image_area or area < max_area_const):
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
    
    # --- NOVO: Table Tail Remover ---
    # Remove "pontinhas" e reflexos finos que ficam grudados na mesa (linha inferior).
    # Desativado para peças ocas (hollow) para evitar que o anel/arco fino seja apagado como se fosse ruído da mesa.
    if not getattr(config, "HOLLOW_SPECIMEN", False):
        has_pixels = np.any(closed == 255, axis=0)
        if np.any(has_pixels):
            y_top = np.argmax(closed == 255, axis=0)
            y_bottom = closed.shape[0] - 1 - np.argmax(closed[::-1, :] == 255, axis=0)
            
            valid_x = np.where(has_pixels)[0]
            valid_y_bottom = y_bottom[valid_x]
            
            # O fundo da imagem pode estar levemente inclinado (câmera torta).
            # Ajustamos uma reta robusta (np.polyfit) ao y_bottom para saber exatamente
            # onde está a mesa para QUALQUER coluna x.
            m, c = np.polyfit(valid_x, valid_y_bottom, 1)
            
            # Avaliar a linha da mesa teórica para todas as colunas
            x_all = np.arange(closed.shape[1])
            local_table_y = m * x_all + c
            
            # Condição 1: A coluna encosta na mesa local (tolerância generosa de 15px para irregularidades da fita)
            touches_table = (local_table_y - y_bottom) <= 15
            
            # Condição 2: A coluna é muito fina (altura < 40 pixels, ~1mm)
            # Reflexos fortes formam "bolhas" que podem ter até 30px de altura.
            is_thin = (y_bottom - y_top) <= 40
            
            # Colunas que são apenas ruído na mesa
            cols_to_erase = has_pixels & touches_table & is_thin
            
            # Apaga o ruído
            closed[:, cols_to_erase] = 0
    
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
    corner_indices = []
    corner_radii_px = []

    if circularity <= 0.90:
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
                corner_indices.append(idx)
                
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
                
            # 4. Calcular raio de curvatura nas quinas
            target_dist_px = perimeter_px * 0.03  # Usar pontos a 3% do perímetro da quina
            n_pts = len(contour)
            for idx in corner_indices:
                # Encontrar ponto antes
                dist_prev = 0
                idx_prev = idx
                while dist_prev < target_dist_px:
                    next_idx = (idx_prev - 1) % n_pts
                    dist_prev += np.linalg.norm(contour[next_idx, 0] - contour[idx_prev, 0])
                    idx_prev = next_idx
                    if dist_prev > target_dist_px * 3: break # fallback
                    
                # Encontrar ponto depois
                dist_next = 0
                idx_next = idx
                while dist_next < target_dist_px:
                    next_idx = (idx_next + 1) % n_pts
                    dist_next += np.linalg.norm(contour[next_idx, 0] - contour[idx_next, 0])
                    idx_next = next_idx
                    if dist_next > target_dist_px * 3: break
                
                p1 = contour[idx_prev, 0]
                p2 = contour[idx, 0]
                p3 = contour[idx_next, 0]
                
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p1 - p3)
                
                s = (a + b + c) / 2
                area_sq = s * (s - a) * (s - b) * (s - c)
                if area_sq > 0:
                    area = np.sqrt(area_sq)
                    radius = (a * b * c) / (4 * area)
                    # Limite de sanidade (raio não pode ser maior que a metade da peça)
                    max_radius = min(rect_w, rect_h)
                    corner_radii_px.append(min(radius, max_radius))
                else:
                    corner_radii_px.append(0.0)
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
        # Geometria (se for retangular)
        "corner_angle_0": corner_angle_0,
        "corner_angle_1": corner_angle_1,
        "corner_angle_2": corner_angle_2,
        "corner_angle_3": corner_angle_3,
        "corner_radius_px": float(np.mean(corner_radii_px)) if corner_radii_px else 0.0,
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
        positions = [i / 100.0 for i in range(0, 101, 5)]

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
    seed_points: list = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Detecta fundo de mesa branco ao redor da placa MDF e remove a borda queimada a laser.
    Também mascara o bloco de calibração se fornecido.
    Homogeniza as regiões externas/bloco com a cor mediana do MDF.
    Novo: Detecta fita azul (chroma key) na borda e mascara a fita e tudo abaixo dela com preto absoluto.
    """
    h_img, w_img = image_color.shape[:2]
    blue_tape_found = False

    # Detectar Fita Azul
    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Limpar a máscara azul
    kernel_blue = np.ones((5,5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_blue)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_blue)
    
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_blue:
        # Pega a fita azul de maior área (para ignorar ruídos soltos)
        c_blue = max(contours_blue, key=cv2.contourArea)
        if cv2.contourArea(c_blue) > 500: # Threshold mínimo de área para ser fita válida
            # Criar uma máscara isolando apenas a fita principal detectada
            main_tape_mask = np.zeros_like(mask_blue)
            cv2.drawContours(main_tape_mask, [c_blue], -1, 255, -1)
            
            # Encontrar a borda superior (y mínimo) para cada coluna x da fita
            has_tape = np.any(main_tape_mask == 255, axis=0)
            x_coords = np.where(has_tape)[0]
            
            if len(x_coords) > 50: # Garantir que temos pontos suficientes para ajustar uma reta
                y_coords = np.argmax(main_tape_mask[:, has_tape] == 255, axis=0)
                
                # Ajustar uma reta (y = mx + c) à borda superior da fita
                m, c = np.polyfit(x_coords, y_coords, 1)
                
                logger.info(f"  [Pre-processamento] Fita azul detectada. Inclinacao: {m:.4f}. Mascarando abaixo da reta.")
                
                # Criar um grid de coordenadas y e x
                y_grid, x_grid = np.mgrid[0:h_img, 0:w_img]
                line_y = m * x_grid + c
                
                # Tudo que estiver na reta ou abaixo dela será mascarado com preto
                below_line_mask = y_grid >= line_y
                
                gray_blurred[below_line_mask] = 0
                image_color[below_line_mask] = [0, 0, 0]
                blue_tape_found = True
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
    
    # Proteger região ao redor do seed_points (evita mascarar peças claras como caulim)
    if seed_points is not None:
        for sp in seed_points:
            protection_radius = max(50, int(min(gray_blurred.shape[:2]) * 0.15))
            cv2.circle(thresh, sp, protection_radius, 0, -1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_blurred, image_color, blue_tape_found
        
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
        
        return gray_out, color_out, blue_tape_found
        
    return gray_blurred, image_color, blue_tape_found


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
    seed_points: list = None,
    mdf_points: list = None,
    shadow_points: list = None,
    hollow: bool = False
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
    gray_blurred, image_color, blue_tape_found = _preprocess_mdf_background(gray_blurred, image_color, calibration_corners, seed_points)

    # Determinar se a peça é mais clara que o fundo usando os seed points
    piece_is_lighter = False
    if seed_points is not None and len(seed_points) > 0 and mdf_points is not None and len(mdf_points) > 0:
        gray_for_brightness = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) if len(image_color.shape) == 3 else gray_blurred
        # Amostrar patch 31x31 ao redor de cada ponto
        patch_r = 15
        h_img, w_img = gray_for_brightness.shape[:2]
        
        piece_brightness_sum = 0
        for sp in seed_points:
            px, py = sp
            y1p = max(0, py - patch_r)
            y2p = min(h_img, py + patch_r + 1)
            x1p = max(0, px - patch_r)
            x2p = min(w_img, px + patch_r + 1)
            piece_brightness_sum += float(np.mean(gray_for_brightness[y1p:y2p, x1p:x2p]))
            
        piece_brightness = piece_brightness_sum / len(seed_points)
        
        mdf_brightness_sum = 0
        for mp in mdf_points:
            mx, my = mp
            y1m = max(0, my - patch_r)
            y2m = min(h_img, my + patch_r + 1)
            x1m = max(0, mx - patch_r)
            x2m = min(w_img, mx + patch_r + 1)
            mdf_brightness_sum += float(np.mean(gray_for_brightness[y1m:y2m, x1m:x2m]))
        
        mdf_brightness = mdf_brightness_sum / len(mdf_points)
        
        piece_is_lighter = piece_brightness > mdf_brightness
        logger.info(
            f"  [Seed] Luminância peça={piece_brightness:.1f}, MDF={mdf_brightness:.1f} "
            f"→ peça {'MAIS CLARA' if piece_is_lighter else 'MAIS ESCURA'} que o fundo"
        )

    mask = None

    if strategy == "auto":
        # Tentar cada estratégia na ordem de preferência
        strategies_to_try = []

        # 1. Se o usuário forneceu sementes
        if seed_points is not None and len(seed_points) > 0 and mdf_points is not None and len(mdf_points) > 0:
            if not blue_tape_found:
                # Vista superior: prioridade total para shadow_band automática
                strategies_to_try.append(("shadow_band", None))

            if blue_tape_found:
                # Com a fita azul garantindo um fundo perfeitamente limpo na vista frontal,
                # o Otsu puro faz uma segmentação infinitamente superior (imune às texturas da peça).
                logger.info("  [Auto] Fita azul detectada: priorizando Otsu puro (ignora texturas/sombras internas da peça).")
                if piece_is_lighter:
                    strategies_to_try.append(("otsu", None))
                    strategies_to_try.append(("lab", None))
                else:
                    strategies_to_try.append(("otsu_dark", None))
                # Fallback para watershed/grabcut apenas se o otsu falhar terrivelmente
                strategies_to_try.append(("watershed_seeded", None))
                strategies_to_try.append(("grabcut_seeded", None))
            else:
                # Sem fita azul (ex: vista superior), o watershed é essencial para não vazar a máscara nas manchas do MDF
                strategies_to_try.append(("watershed_seeded", None))
                strategies_to_try.append(("grabcut_seeded", None))

        # 2. Se sabemos conclusivamente a cor relativa da peça pelas sementes,
        # e watershed falhou ou não estava disponível, tentamos thresholds globais.
        if piece_is_lighter:
            strategies_to_try.append(("otsu", None))
            strategies_to_try.append(("lab", None))
        else:
            strategies_to_try.append(("otsu_dark", None))

        # 3. Estratégia dedicada para argila (útil se as peças são muito escuras ou soltam muito pó)
        if getattr(config, "MATERIAL_TYPE", "Argila") == "Argila":
            strategies_to_try.append(("edges", None))

        # 4. Background subtraction
        if background is not None:
            strategies_to_try.append(("background_sub", background))
        
        # 5. Fallbacks finais
        if not piece_is_lighter:
            strategies_to_try.append(("otsu", None))
        else:
            strategies_to_try.append(("otsu_dark", None))

        for strat_name, bg in strategies_to_try:
            logger.debug(f"  Auto: tentando '{strat_name}'...")
            try:
                candidate_mask = _apply_strategy(
                    strat_name, gray_blurred, image_color, bg,
                    piece_is_lighter=piece_is_lighter,
                    seed_points=seed_points,
                    mdf_points=mdf_points,
                    calibration_corners=calibration_corners,
                    shadow_points=shadow_points
                )
                candidate_mask = check_and_correct_inversion(candidate_mask)
                candidate_mask = postprocess_mask(candidate_mask)
                if hollow:
                    candidate_mask = fill_hollow_mask(candidate_mask)
                quality = evaluate_mask_quality(candidate_mask)

                logger.debug(
                    f"  Auto '{strat_name}': proporção={quality['proportion']:.3f}, "
                    f"contornos={quality['n_contours']}, "
                    f"compacidade={quality['compactness']:.3f}, "
                    f"ok={quality['is_good']}"
                )

                if quality["is_good"]:
                    # Verificar se pelo menos um contorno válido contém um seed point.
                    # Sem esta verificação, uma máscara pode ter boa qualidade geral
                    # mas não capturar a peça que o usuário marcou — gerando um dummy.
                    if seed_points is not None and len(seed_points) > 0:
                        cand_contours, _ = cv2.findContours(
                            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        seed_found_in_contour = False
                        
                        min_area = getattr(config, 'MIN_CONTOUR_AREA_PX', 5000)
                        if hollow:
                            min_area = 500
                            
                        for c in cand_contours:
                            if cv2.contourArea(c) < min_area:
                                continue
                            for sp in seed_points:
                                if cv2.pointPolygonTest(c, (float(sp[0]), float(sp[1])), False) >= 0:
                                    seed_found_in_contour = True
                                    break
                            if seed_found_in_contour:
                                break
                        if not seed_found_in_contour:
                            logger.info(
                                f"  ✗ Estratégia '{strat_name}' passou qualidade mas nenhum contorno "
                                f"contém um seed point. Descartando."
                            )
                            continue

                    mask = candidate_mask
                    logger.info(f"  ✓ Estratégia selecionada: '{strat_name}'")
                    strategy = strat_name
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
            seed_points=seed_points,
            mdf_points=mdf_points,
            calibration_corners=calibration_corners,
            shadow_points=shadow_points
        )
        mask = check_and_correct_inversion(mask)
        mask = postprocess_mask(mask)
        if hollow:
            mask = fill_hollow_mask(mask)

    # A separação forçada foi removida pois o algoritmo Watershed já separa 
    # naturalmente as peças (colocando bordas de -1 entre bacias distintas). 
    # Os cortes com cv2.line estavam fatiando as peças incorretamente.

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
        
        max_prop = 0.95 if hollow else config.MAX_CONTOUR_AREA_PROPORTION
        
        # Tentar carregar resolução da calibração para normalizar os limites de área
        full_image_area = total_pixels
        try:
            import os
            if os.path.exists(config.CALIBRATION_FILE):
                fs = cv2.FileStorage(config.CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
                w_node = fs.getNode("image_width")
                h_node = fs.getNode("image_height")
                if not w_node.empty() and not h_node.empty():
                    full_image_area = int(w_node.real()) * int(h_node.real())
                fs.release()
        except Exception:
            pass

        resolution_factor = max(1.0, full_image_area / 10000000.0)
        max_area_const = 150000 * resolution_factor
        
        if circularity < 0.05 or aspect_ratio > 10.0 or (area >= max_prop * full_image_area and area >= max_area_const):
            logger.info(
                f"  Descartando contorno ruidoso/muito grande: área={area:.0f} px², "
                f"bbox={w}x{h} px, circularidade={circularity:.3f}, aspect_ratio={aspect_ratio:.2f}"
            )
            continue
            
        valid_contours.append(c)

    # Refinar contornos válidos usando as bordas Canny
    refined_contours = []
    for c in valid_contours:
        # Se a fita azul foi usada, a máscara (já filtrada pelo Table Tail Remover)
        # é infinitamente mais precisa que o Canny. O Canny iria detectar a borda
        # física do reflexo da fita e recolocar o ruído que acabamos de apagar!
        if blue_tape_found:
            logger.info("    [Refinamento Canny] Ignorado pois a máscara (Otsu + Tail Remover) já possui precisão sub-pixel.")
            refined_contours.append(c)
            continue
            
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

    # Se for uma peça oca/modo vaso, convertemos os contornos para convexHull para garantir o preenchimento da área
    if hollow and len(valid_contours) > 0:
        logger.info("  [Hollow] Convertendo contornos para Convex Hull como garantia de fechamento")
        valid_contours = [cv2.convexHull(c) for c in valid_contours]

    # Filtrar contornos pelo seed_points: se disponível, reter apenas os que contêm sementes
    if seed_points is not None and len(seed_points) > 0:
        containing = []
        for c in valid_contours:
            contains = False
            for sp in seed_points:
                if cv2.pointPolygonTest(c, (float(sp[0]), float(sp[1])), False) >= 0:
                    contains = True
                    break
            if contains:
                containing.append(c)
        if containing:
            logger.info(f"  [Seed] {len(containing)} contorno(s) contém ponto-semente. Retendo apenas eles.")
            valid_contours = containing

    # Ordenar contornos: se tivermos seed_points, ordenamos os contornos na EXATA ORDEM das sementes!
    if seed_points is not None and len(seed_points) > 0:
        ordered_contours = []
        for sp in seed_points:
            best_c = None
            best_dist = float('inf')
            sp_x, sp_y = float(sp[0]), float(sp[1])
            
            # 1. Tentar achar um contorno que de fato contenha a semente
            containing_contours = []
            for c in valid_contours:
                if cv2.pointPolygonTest(c, (sp_x, sp_y), False) >= 0:
                    containing_contours.append(c)
                    
            if containing_contours:
                best_c = max(containing_contours, key=cv2.contourArea)
            else:
                # Fallback: pegar o contorno mais próximo usando pointPolygonTest
                for c in valid_contours:
                    dist = abs(cv2.pointPolygonTest(c, (sp_x, sp_y), True))
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c
            
            # Se não achou ou está muito distante, gerar dummy
            if best_c is None or (best_dist > 300 and not containing_contours):
                logger.warning(f"  [Seed] Semente ({sp_x}, {sp_y}) sem contorno próximo. Gerando contorno dummy.")
                best_c = create_dummy_contour(int(sp_x), int(sp_y))
                
            ordered_contours.append(best_c)
            
        valid_contours = ordered_contours
        logger.info(f"  Contornos ordenados restritamente pela ordem das {len(seed_points)} sementes do usuário (Total: {len(valid_contours)}).")

    elif len(valid_contours) > 0:
        # Fallback para o comportamento padrão sem sementes
        h, w = mask.shape[:2]
        cx_img, cy_img = w // 2, h // 2
        max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)
        ref_x = float(cx_img)
        ref_y = float(cy_img)
        
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

    # Suavização de contorno específica para shadow_band
    if strategy == "shadow_band" and len(valid_contours) > 0:
        logger.info("    [Shadow Band] Aplicando suavização de contorno (moving average)...")
        win_size = getattr(config, "SHADOW_BAND_SMOOTH_WINDOW", 15)
        valid_contours = [smooth_contour(c, window_size=win_size) for c in valid_contours]
        
        if getattr(config, "SHADOW_BAND_CONVEX_HULL", True):
            logger.info("    [Shadow Band] Aplicando Convex Hull para fechar reentrâncias...")
            valid_contours = [cv2.convexHull(c) for c in valid_contours]

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


def create_dummy_contour(cx: int, cy: int) -> np.ndarray:
    """Cria um contorno retangular dummy de 10x10 pixels centralizado em (cx, cy)."""
    return np.array([
        [[cx - 5, cy - 5]],
        [[cx + 5, cy - 5]],
        [[cx + 5, cy + 5]],
        [[cx - 5, cy + 5]]
    ], dtype=np.int32)


def segment_seeded_threshold(
    image: np.ndarray,
    seed_points: list | tuple,
    mdf_points: list,
    piece_is_lighter: bool = None,
    max_dim: int = 1200
) -> np.ndarray:
    """
    Segmentação por threshold dinâmico com base em sementes de cor do objeto e MDF.
    Pode receber imagem BGR ou cinza, e sementes em formato de lista ou tupla única.
    """
    if isinstance(seed_points, tuple):
        seed_points = [seed_points]
        
    h_orig, w_orig = image.shape[:2]
    
    # Calcular escala de redimensionamento
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        
    if scale < 1.0:
        img_small = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sps = [(int(sp[0] * scale), int(sp[1] * scale)) for sp in seed_points]
        mp = (int(mdf_points[0] * scale), int(mdf_points[1] * scale))
    else:
        img_small = image.copy()
        sps = seed_points
        mp = mdf_points
        
    h_small, w_small = img_small.shape[:2]
    
    # Determinar se a imagem é colorida
    is_color = len(img_small.shape) == 3 and img_small.shape[2] == 3
    
    if is_color:
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        channels = [gray, l_ch, a_ch, b_ch]
        channel_names = ["Cinza", "LAB_L", "LAB_A", "LAB_B"]
    else:
        gray = img_small
        channels = [gray]
        channel_names = ["Cinza"]
        
    # Escolher o melhor canal comparando a diferença entre as sementes da peça e do MDF
    best_channel_idx = 0
    best_diff = -1.0
    best_threshold = 127
    best_piece_is_lighter = True
    
    # Ajustar raio de amostragem proporcionalmente ao tamanho reduzido
    patch_r = max(3, int(min(h_small, w_small) * 0.01))
    
    for ch_idx, ch in enumerate(channels):
        piece_vals = []
        for sp in sps:
            px, py = sp
            y1 = max(0, py - patch_r)
            y2 = min(h_small, py + patch_r + 1)
            x1 = max(0, px - patch_r)
            x2 = min(w_small, px + patch_r + 1)
            piece_vals.append(np.mean(ch[y1:y2, x1:x2]))
        piece_avg = np.mean(piece_vals)
        
        mx, my = mp
        y1 = max(0, my - patch_r)
        y2 = min(h_small, my + patch_r + 1)
        x1 = max(0, mx - patch_r)
        x2 = min(w_small, mx + patch_r + 1)
        mdf_avg = np.mean(ch[y1:y2, x1:x2])
        
        diff = abs(piece_avg - mdf_avg)
        if diff > best_diff:
            best_diff = diff
            best_channel_idx = ch_idx
            best_threshold = (piece_avg + mdf_avg) / 2.0
            best_piece_is_lighter = piece_avg > mdf_avg
            
    ch_best = channels[best_channel_idx]
    
    # Se o chamador especificou piece_is_lighter, respeitar
    if piece_is_lighter is not None:
        best_piece_is_lighter = piece_is_lighter
        
    logger.info(f"  [Seeded Threshold] Canal: {channel_names[best_channel_idx]} (contraste: {best_diff:.1f}, limiar: {best_threshold:.1f}, peça_mais_clara: {best_piece_is_lighter})")
    
    if best_diff < 5.0:
        logger.warning("  [Seeded Threshold] Contraste muito baixo. Usando Otsu de fallback.")
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if best_piece_is_lighter:
            _, thresh = cv2.threshold(ch_best, int(best_threshold), 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(ch_best, int(best_threshold), 255, cv2.THRESH_BINARY_INV)
            
    # Fechamento morfológico para fechar as ranhuras internas da peça
    close_sz = max(15, int(min(h_small, w_small) * 0.025))
    if close_sz % 2 == 0:
        close_sz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar os contornos da máscara binária
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = max(100, int(config.MIN_CONTOUR_AREA_PX * (scale ** 2) / 4))
    valid_candidates = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_candidates:
        valid_candidates = contours
        
    # Criar uma máscara final contendo APENAS os contornos que estão associados às sementes
    mask_small = np.zeros_like(thresh)
    for sp in sps:
        sp_pt = (float(sp[0]), float(sp[1]))
        best_c = None
        best_score = -float('inf')
        
        for c in valid_candidates:
            score = cv2.pointPolygonTest(c, sp_pt, True)
            if score > best_score:
                best_score = score
                best_c = c
                
        if best_c is not None:
            cv2.drawContours(mask_small, [best_c], -1, 255, -1)
            
    # Upsampling da máscara
    if scale < 1.0:
        mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        blur_size = max(3, int(1 / scale) * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = mask_small
        
    return mask


def segment_watershed_seeded(
    image_color: np.ndarray,
    seed_points: list,
    mdf_points: list,
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
    # Para peças ocas (modo vaso), usamos uma resolução muito maior (ou total se ROI < 4000px)
    # para evitar que a parede fina seja reduzida a poucos pixels e as sementes vazem.
    local_max_dim = 4000 if getattr(config, "HOLLOW_SPECIMEN", False) else max_dim
    scale = 1.0
    if max(h_orig, w_orig) > local_max_dim:
        scale = local_max_dim / max(h_orig, w_orig)
    
    if scale < 1.0:
        img_small = cv2.resize(image_color, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = image_color.copy()
        
    h_small, w_small = img_small.shape[:2]
    
    # Bilateral Filter: suaviza a granulação do MDF sem destruir (borrar) as bordas físicas!
    # Isso impede que o gradiente da peça se expanda para a sombra.
    img_small = cv2.bilateralFilter(img_small, 9, 75, 75)
    
    mps = [(int(p[0] * scale), int(p[1] * scale)) for p in mdf_points]
    
    # Sementes redimensionadas: para peças ocas (paredes finas), a semente de foreground
    # deve ser muito pequena (ex: raio 2) para não vazar da parede fina para o MDF de fundo.
    if getattr(config, "HOLLOW_SPECIMEN", False):
        fg_seed_radius = 2
        bg_seed_radius = 5  # Muito pequeno também para evitar que cliques no MDF pertos do anel invadam a parede
    else:
        # Reduzido de min * 0.015 (~10px) para 3px para evitar vazamentos ao clicar perto das bordas.
        fg_seed_radius = 3
        bg_seed_radius = 6
    
    # Marcadores para o Watershed:
    # 0 = Desconhecido (onde o algoritmo vai decidir)
    # 1 a N = Foreground (Peças independentes, para forçar separação entre elas)
    # N+1 = Background garantido (MDF e bordas)
    markers = np.zeros((h_small, w_small), dtype=np.int32)
    bg_marker = len(seed_points) + 1
    
    # Marcar foreground (IDs únicos) nos pontos das peças
    for i, sp in enumerate(seed_points):
        s_pt = (int(sp[0] * scale), int(sp[1] * scale))
        cv2.circle(markers, s_pt, fg_seed_radius, i + 1, -1)
    
    # Marcar background no ponto do MDF
    for mp in mps:
        cv2.circle(markers, mp, bg_seed_radius, bg_marker, -1)
    
    # Marcar bordas da imagem como background garantido
    # Se for peça oca (modo vaso), usamos apenas uma borda fina de 2 pixels
    # para evitar que a semente ou a parede da peça sejam 'comidas' pela borda do background se a ROI for estreita.
    if getattr(config, "HOLLOW_SPECIMEN", False):
        border_w = 2
        border_h = 2
    else:
        border_w = max(5, int(w_small * 0.02))
        border_h = max(5, int(h_small * 0.02))
        
    markers[:border_h, :] = bg_marker
    markers[-border_h:, :] = bg_marker
    markers[:, :border_w] = bg_marker
    markers[:, -border_w:] = bg_marker
    
    # Marcar bloco de calibração como background garantido (2)
    if calibration_corners is not None:
        try:
            hull = cv2.convexHull(calibration_corners.astype(np.int32))
            if scale < 1.0:
                hull = (hull * scale).astype(np.int32)
            
            calib_mask = np.zeros((h_small, w_small), dtype=np.uint8)
            cv2.drawContours(calib_mask, [hull], -1, 255, -1)
            
            # Dilatar para cobrir a borda branca do bloco
            dilation_px = max(15, int(min(h_small, w_small) * 0.04))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
            calib_mask = cv2.dilate(calib_mask, kernel)
            
            markers[calib_mask == 255] = bg_marker
        except Exception as e:
            logger.warning(f"  Watershed: Erro ao marcar calibração: {e}")
            
    # Executar Watershed (modifica a matriz 'markers' inplace)
    cv2.watershed(img_small, markers)
    
    # Criar máscara binária apenas das áreas classificadas como Foreground (IDs 1 a N)
    mask_small = np.zeros((h_small, w_small), dtype=np.uint8)
    for i in range(len(seed_points)):
        mask_small[markers == (i + 1)] = 255
    
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

def smooth_contour(contour: np.ndarray, window_size: int = 31) -> np.ndarray:
    """
    Aplica um filtro de média móvel com padding circular nos pontos do contorno
    para suavizar e remover irregularidades/pontas.
    """
    if len(contour) < window_size:
        return contour
    
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    
    pad = window_size // 2
    x_padded = np.concatenate([x[-pad:], x, x[:pad]])
    y_padded = np.concatenate([y[-pad:], y, y[:pad]])
    
    kernel = np.ones(window_size) / window_size
    x_smooth = np.convolve(x_padded, kernel, mode='valid')
    y_smooth = np.convolve(y_padded, kernel, mode='valid')
    
    smoothed = np.zeros_like(contour)
    smoothed[:, 0, 0] = np.round(x_smooth)
    smoothed[:, 0, 1] = np.round(y_smooth)
    return smoothed.astype(np.int32)


def segment_shadow_band(
    image_color: np.ndarray,
    seed_points: list,
    mdf_points: list,
    shadow_points: list = None,
    calibration_corners: np.ndarray = None
) -> np.ndarray:
    """
    Segmentação por rejeição de banda em relação ao MDF, baseada em sementes de cor.
    Qualquer pixel significativamente mais claro (peça) ou mais escuro (sombra)
    que o MDF é considerado objeto.
    A faixa é calculada de forma adaptativa a partir das intensidades médias
    amostradas nos seed_points, mdf_points e opcionalmente shadow_points.
    """
    h_orig, w_orig = image_color.shape[:2]
    
    # Converter para escala de cinza e aplicar desfoque suave para reduzir ruído de textura
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) if len(image_color.shape) == 3 else image_color.copy()
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Amostrar os valores de cinza nos pontos clicados pelo usuário
    # Usamos uma janela pequena 5x5 ao redor de cada ponto para robustez contra ruído de pixel único
    patch_r = 2
    
    def get_mean_val(pts):
        vals = []
        for pt in pts:
            px, py = int(pt[0]), int(pt[1])
            y1 = max(0, py - patch_r)
            y2 = min(h_orig, py + patch_r + 1)
            x1 = max(0, px - patch_r)
            x2 = min(w_orig, px + patch_r + 1)
            vals.append(np.mean(blurred[y1:y2, x1:x2]))
        return np.mean(vals)
        
    v_mdf = get_mean_val(mdf_points)
    v_piece = get_mean_val(seed_points)
    
    # Calcular limiares de corte
    if shadow_points is not None and len(shadow_points) > 0:
        v_shadow = get_mean_val(shadow_points)
        t_shadow_max = (v_mdf + v_shadow) / 2.0
        logger.info(f"  [Shadow Band] Amostras de cinza -> MDF: {v_mdf:.1f}, Peca: {v_piece:.1f}, Sombra (Manual): {v_shadow:.1f}")
    else:
        delta_low = getattr(config, "SHADOW_BAND_DELTA_LOW", 30)
        t_shadow_max = v_mdf - delta_low
        logger.info(f"  [Shadow Band] Amostras de cinza -> MDF: {v_mdf:.1f}, Peca: {v_piece:.1f}. Sombra (Auto): <= {t_shadow_max:.1f} (MDF - {delta_low})")
        
    t_piece_min = (v_mdf + v_piece) / 2.0
    
    # Adicionar uma margem de segurança caso a diferença seja muito pequena
    t_shadow_max = min(t_shadow_max, v_mdf - 15)
    t_piece_min = max(t_piece_min, v_mdf + 15)
    
    logger.info(f"  [Shadow Band] Limiares calculados -> Sombra <= {t_shadow_max:.1f}, Peca >= {t_piece_min:.1f}")
    
    # Limiar mínimo de cinza para ignorar a mesa preta externa (que tem cinza ~40)
    min_gray = getattr(config, "SHADOW_BAND_MIN_GRAY", 70)
    
    # Criar máscara binária
    mask_piece = (blurred >= t_piece_min)
    mask_shadow = (blurred <= t_shadow_max) & (blurred > min_gray)
    
    mask = np.zeros_like(gray)
    mask[mask_piece | mask_shadow] = 255
    
    # 1. Abertura morfológica para eliminar ruídos isolados (pequenos pontos no MDF)
    open_size = 3 if getattr(config, "HOLLOW_SPECIMEN", False) else 5
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Fechamento morfológico para conectar a peça e a sombra (usando kernel 45x45 para evitar reentrâncias nas pontas)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Erosão física da máscara para ajustar o contorno mais próximo da peça física
    erosion_mm = getattr(config, "SHADOW_BAND_EROSION_MM", 0.5)
    px_per_mm = 1.0
    if calibration_corners is not None and len(calibration_corners) > 0:
        import metrology
        try:
            scale = metrology.calibrate_scale_from_block(calibration_corners)
            px_per_mm = (scale["px_per_mm_h"] + scale["px_per_mm_v"]) / 2.0
        except Exception:
            pass
    scale_val = px_per_mm if px_per_mm > 1.0 else 17.0

    # 3a. Erosão Simétrica
    if erosion_mm > 0:
        erosion_px = erosion_mm * scale_val
        k_size = int(round(2 * erosion_px + 1))
        if k_size % 2 == 0:
            k_size += 1
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask_closed = cv2.erode(mask_closed, kernel_erode)
        logger.info(f"  [Shadow Band] Aplicada erosão simétrica de {erosion_mm} mm ({erosion_px:.1f} px, kernel {k_size}x{k_size})")

    # 3b. Erosão Assimétrica na Base (Deslocamento para cima da borda inferior)
    bottom_shift_mm = getattr(config, "SHADOW_BAND_BOTTOM_SHIFT_MM", 0.0)
    if bottom_shift_mm > 0:
        shift_y_px = int(round(bottom_shift_mm * scale_val))
        if shift_y_px > 0:
            shifted = np.zeros_like(mask_closed)
            shifted[:-shift_y_px, :] = mask_closed[shift_y_px:, :]
            mask_closed = cv2.bitwise_and(mask_closed, shifted)
            logger.info(f"  [Shadow Band] Aplicada erosão assimétrica na base de {bottom_shift_mm} mm ({shift_y_px} px)")
        
    return mask_closed


def _apply_strategy(
    strategy: str,
    gray_blurred: np.ndarray,
    image_color: np.ndarray,
    background: np.ndarray = None,
    piece_is_lighter: bool = False,
    seed_points: list = None,
    mdf_points: list = None,
    calibration_corners: np.ndarray = None,
    shadow_points: list = None
) -> np.ndarray:
    """Aplica uma estratégia de segmentação específica."""
    if strategy == "shadow_band":
        if seed_points is None or len(seed_points) == 0 or mdf_points is None or len(mdf_points) == 0:
            raise ValueError("Estratégia 'shadow_band' requer seed_points e mdf_points.")
        return segment_shadow_band(
            image_color, seed_points, mdf_points, shadow_points,
            calibration_corners=calibration_corners
        )

    elif strategy == "watershed_seeded":
        if seed_points is None or len(seed_points) == 0 or mdf_points is None or len(mdf_points) == 0:
            raise ValueError("Estratégia 'watershed_seeded' requer seed_points e mdf_points.")
        return segment_watershed_seeded(
            image_color, seed_points, mdf_points,
            calibration_corners=calibration_corners
        )

    elif strategy == "grabcut_seeded":
        if seed_points is None or len(seed_points) == 0 or mdf_points is None or len(mdf_points) == 0:
            raise ValueError("Estratégia 'grabcut_seeded' requer seed_points e mdf_points.")
        return segment_grabcut_seeded(
            image_color, seed_points, mdf_points,
            iterations=5, max_dim=1000,
            calibration_corners=calibration_corners
        )

    elif strategy == "edges":
        return segment_edges(gray_blurred, seed_points, calibration_corners)

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
