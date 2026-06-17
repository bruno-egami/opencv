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
        # Contorno original (para desenho e conversão)
        "contour": contour,
        "hull": hull,
    }

    return metrics


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
    masks_dir: str = None
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

    mask = None

    if strategy == "auto":
        # Tentar cada estratégia na ordem de preferência
        strategies_to_try = []

        strategies_to_try.append(("lab_b", None))
        if background is not None:
            strategies_to_try.append(("background_sub", background))
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
        
        if circularity < 0.05 or aspect_ratio > 5.0:
            logger.info(
                f"  Descartando contorno ruidoso: área={area:.0f} px², "
                f"bbox={w}x{h} px, circularidade={circularity:.3f}, aspect_ratio={aspect_ratio:.2f}"
            )
            continue
            
        valid_contours.append(c)

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
