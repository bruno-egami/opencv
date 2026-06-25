# -*- coding: utf-8 -*-
"""
Módulo de metrologia: calibração de escala via bloco padrão coplanar.

Responsável por:
1. Detectar o padrão xadrez (chessboard) do bloco padrão de calibração
2. Calcular fatores de escala px_per_mm independentes para H e V a partir do bloco
3. Converter todas as métricas de pixels para milímetros
"""

import logging
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class MetrologyError(Exception):
    """Exceção para erros na calibração de escala."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Detecção do Bloco Padrão
# ──────────────────────────────────────────────────────────────────────────────

def detect_calibration_block(
    image: np.ndarray,
    pattern_size: tuple = None,
    square_size_mm: float = None
) -> np.ndarray:
    """
    Detecta automaticamente o padrão xadrez (chessboard) do bloco padrão de calibração.

    Tenta encontrar os cantos internos usando cv2.findChessboardCorners.
    Se falhar na imagem original, tenta redimensionar a imagem (downscale)
    para maior robustez em imagens de alta resolução com ruído.
    Aplica cv2.cornerSubPix para precisão subpixel.

    Args:
        image: Imagem de entrada (BGR ou Gray, undistorted).
        pattern_size: Tupla (colunas_internas, linhas_internas) de cantos.
                      Default: config.CALIB_BLOCK_PATTERN_SIZE
        square_size_mm: Tamanho real de cada quadrado (não usado aqui, mas para assinatura).

    Returns:
        Array Nx1x2 de pontos (x, y) de cantos refinados, ou None se falhar.
    """
    if pattern_size is None:
        pattern_size = getattr(config, "CALIB_BLOCK_PATTERN_SIZE", (8, 7))

    cols, rows = pattern_size
    transposed_size = (rows, cols)

    logger.info(f"Detectando bloco padrão de calibração (tamanho do padrão: {pattern_size})...")

    # Converter para escala de cinza se colorida
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Flags com e sem FAST_CHECK
    flags_fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    flags_no_fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret = False
    corners = None

    # Testar diferentes resoluções (escalas) ordenadas da menor para a maior para velocidade
    # Escalas menores (0.1, 0.15) rodam instantaneamente e filtram ruído de alta frequência
    for scale in [0.1, 0.15, 0.25, 0.5, 1.0]:
        if scale == 1.0:
            gray_sc = gray
        else:
            gray_sc = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Evitar erros de assert se a imagem for pequena demais
        min_dim = max(cols, rows) * 6
        if gray_sc.shape[1] < min_dim or gray_sc.shape[0] < min_dim:
            continue

        # 1. Tentar padrão original
        ret_orig, corners_sc = cv2.findChessboardCorners(gray_sc, pattern_size, flags=flags_fast)
        if not ret_orig:
            ret_orig, corners_sc = cv2.findChessboardCorners(gray_sc, pattern_size, flags=flags_no_fast)

        if ret_orig:
            logger.info(f"[OK] Bloco padrao detectado com escala {scale}")
            corners = corners_sc / scale if scale != 1.0 else corners_sc
            ret = True
            break

        # 2. Tentar padrão transposto (rotacionado)
        if transposed_size != pattern_size:
            ret_trans, corners_sc = cv2.findChessboardCorners(gray_sc, transposed_size, flags=flags_fast)
            if not ret_trans:
                ret_trans, corners_sc = cv2.findChessboardCorners(gray_sc, transposed_size, flags=flags_no_fast)

            if ret_trans:
                logger.info(f"[OK] Bloco padrao (rotacionado) detectado com escala {scale}")
                # Transpor os cantos detectados para a orientação padrão (cols * rows, 1, 2)
                c_grid = corners_sc.reshape(cols, rows, 2)
                c_transposed = c_grid.transpose(1, 0, 2)
                corners_orig_sc = c_transposed.reshape(-1, 1, 2)

                corners = corners_orig_sc / scale if scale != 1.0 else corners_orig_sc
                ret = True
                break

    # --- INÍCIO DO AJUSTE MANUAL INTERATIVO ---
    import sys
    is_testing = "pytest" in sys.modules
    use_interactive = getattr(config, "INTERACTIVE_CALIBRATION", False) and not is_testing

    if use_interactive:
        import interactive
        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        initial_points = None
        if ret and corners is not None:
            corners_reshaped = corners.reshape(-1, 2)
            tuned_points, confirmed = interactive.fine_tune_checkerboard_grid(
                img_color,
                corners_reshaped,
                pattern_size=pattern_size,
                window_title="Validar Malha do Bloco Padrao"
            )
            if confirmed:
                corners = tuned_points.reshape(-1, 1, 2)
                ret = True
        else:
            selected = interactive.select_checkerboard_corners_manually(
                img_color,
                window_title="Falha Automatica: Selecionar 4 Cantos do Bloco Padrao"
            )
            if selected is not None:
                src_pts = np.array([
                    [0, 0],
                    [cols - 1, 0],
                    [cols - 1, rows - 1],
                    [0, rows - 1]
                ], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, selected)
                grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
                ideal_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
                projected_grid = cv2.perspectiveTransform(ideal_grid.reshape(-1, 1, 2), H)
                initial_points = projected_grid.reshape(-1, 2)
                
                tuned_points, confirmed = interactive.fine_tune_checkerboard_grid(
                    img_color,
                    initial_points,
                    pattern_size=pattern_size,
                    window_title="Ajuste Fino da Malha do Bloco Padrao"
                )
                if confirmed:
                    corners = tuned_points.reshape(-1, 1, 2)
                    ret = True
                else:
                    corners = None
                    ret = False

    if not ret or corners is None:
        logger.warning("Bloco padrão de calibração não foi detectado automaticamente.")
        return None

    # Refinamento subpixel na imagem original
    # Usar janela que escala com a resolução da imagem para alta precisão
    win_size = max(5, int(gray.shape[1] / 1000))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = corners.astype(np.float32)
    cv2.cornerSubPix(gray, corners, (win_size, win_size), (-1, -1), criteria)

    logger.info(f"[OK] Bloco padrão de calibração detectado e refinado subpixel: {len(corners)} cantos")
    return corners


def calibrate_scale_from_block(
    corners: np.ndarray,
    pattern_size: tuple = None,
    square_size_mm: float = None
) -> dict:
    """
    Calcula a escala px_per_mm horizontal e vertical a partir dos cantos detectados no bloco padrão.

    Calcula a distância média em pixels entre cantos adjacentes nas linhas (horizontal)
    e nas colunas (vertical), e divide pelo square_size_mm para obter px_per_mm.

    Os cantos são organizados por cv2.findChessboardCorners na ordem:
    esquerda-para-direita, cima-para-baixo.

    Args:
        corners: Array Nx1x2 (ou Nx2) com os cantos internos validados (em pixels).
        pattern_size: Tupla (colunas_internas, linhas_internas) de cantos.
                      Default: config.CALIB_BLOCK_PATTERN_SIZE (8, 7)
        square_size_mm: Tamanho real do lado de cada quadrado em mm.
                        Default: config.CALIB_BLOCK_SQUARE_SIZE_MM (6.0)

    Returns:
        Dict com a escala nos eixos H e V:
            - px_per_mm_h: Fator horizontal (pixels por mm)
            - px_per_mm_v: Fator vertical (pixels por mm)
            - anisotropy: Diferença relativa (|H - V| / max(H, V))
            - linearity_h: Desvio padrão relativo dos espaçamentos horizontais
            - linearity_v: Desvio padrão relativo dos espaçamentos verticais
            - n_points: Número de cantos usados
            - view_mode: "top"
    """
    if pattern_size is None:
        pattern_size = getattr(config, "CALIB_BLOCK_PATTERN_SIZE", (8, 7))
    if square_size_mm is None:
        square_size_mm = getattr(config, "CALIB_BLOCK_SQUARE_SIZE_MM", 6.0)

    cols, rows = pattern_size
    pts = corners.reshape(-1, 2)

    if len(pts) != cols * rows:
        raise MetrologyError(
            f"Número incorreto de cantos para calibração. Esperado {cols * rows}, obtido {len(pts)}."
        )

    # 1. Calcular espaçamentos horizontais (entre colunas adjacentes na mesma linha)
    h_spacings = []
    for r in range(rows):
        row_idx_start = r * cols
        for c in range(cols - 1):
            pt_left = pts[row_idx_start + c]
            pt_right = pts[row_idx_start + c + 1]
            dist = np.linalg.norm(pt_right - pt_left)
            h_spacings.append(dist)

    # 2. Calcular espaçamentos verticais (entre linhas adjacentes na mesma coluna)
    v_spacings = []
    for c in range(cols):
        for r in range(rows - 1):
            pt_top = pts[r * cols + c]
            pt_bottom = pts[(r + 1) * cols + c]
            dist = np.linalg.norm(pt_bottom - pt_top)
            v_spacings.append(dist)

    h_spacings = np.array(h_spacings)
    v_spacings = np.array(v_spacings)

    # Filtrar outliers
    h_spacings = _filter_outliers(h_spacings)
    v_spacings = _filter_outliers(v_spacings)

    mean_h_px = np.mean(h_spacings)
    mean_v_px = np.mean(v_spacings)

    # O bloco está no mesmo plano da face a ser medida, então a escala é direta!
    px_per_mm_h = mean_h_px / square_size_mm
    px_per_mm_v = mean_v_px / square_size_mm

    anisotropy = abs(px_per_mm_h - px_per_mm_v) / max(px_per_mm_h, px_per_mm_v)
    linearity_h = np.std(h_spacings) / mean_h_px if mean_h_px > 0 else 0.0
    linearity_v = np.std(v_spacings) / mean_v_px if mean_v_px > 0 else 0.0

    logger.info("Calibração via Bloco Padrão:")
    logger.info(f"  Espaçamento horizontal médio: {mean_h_px:.2f} px ({square_size_mm} mm)")
    logger.info(f"  Espaçamento vertical médio:   {mean_v_px:.2f} px ({square_size_mm} mm)")
    logger.info(f"  px_per_mm_h: {px_per_mm_h:.4f}")
    logger.info(f"  px_per_mm_v: {px_per_mm_v:.4f}")
    logger.info(f"  Anisotropia: {anisotropy:.1%}")

    return {
        "px_per_mm_h": px_per_mm_h,
        "px_per_mm_v": px_per_mm_v,
        "anisotropy": anisotropy,
        "linearity_h": linearity_h,
        "linearity_v": linearity_v,
        "n_points": len(pts),
        "view_mode": "top"
    }


def _filter_outliers(values: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Remove outliers usando IQR (Interquartile Range)."""
    if len(values) < 4:
        return values

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    filtered = values[(values >= lower) & (values <= upper)]
    n_removed = len(values) - len(filtered)

    if n_removed > 0:
        logger.debug(f"  Outliers removidos: {n_removed}/{len(values)}")

    return filtered if len(filtered) > 0 else values


def convert_measurements(metrics_px: dict, scale: dict) -> dict:
    """
    Converte métricas de pixels para milímetros usando os fatores de escala.

    Utiliza fatores independentes para H e V para lidar com anisotropia.

    Args:
        metrics_px: Dict com métricas em pixels (de segmentation.extract_contour_metrics).
        scale: Dict com fatores de escala.

    Returns:
        Dict com métricas originais + métricas em mm adicionadas.
    """
    px_h = scale["px_per_mm_h"]
    px_v = scale["px_per_mm_v"]
    px_avg = (px_h + px_v) / 2

    metrics_mm = dict(metrics_px)  # Copiar originais

    # Bounding box
    metrics_mm["bbox_w_mm"] = metrics_px["bbox_w"] / px_h
    metrics_mm["bbox_h_mm"] = metrics_px["bbox_h"] / px_v

    # Retângulo mínimo: decompõe os fatores horizontal/vertical com base no ângulo de rotação
    angle_deg = metrics_px.get("min_rect_angle", 0.0)
    theta = np.radians(angle_deg)

    # Fator de escala para min_rect_w (orientado em theta)
    denom_w = (np.cos(theta) / px_h) ** 2 + (np.sin(theta) / px_v) ** 2
    px_w = 1.0 / np.sqrt(denom_w) if denom_w > 0 else px_avg

    # Fator de escala para min_rect_h (orientado em theta + pi/2)
    denom_h = (np.sin(theta) / px_h) ** 2 + (np.cos(theta) / px_v) ** 2
    px_h_rect = 1.0 / np.sqrt(denom_h) if denom_h > 0 else px_avg

    metrics_mm["min_rect_w_mm"] = metrics_px["min_rect_w"] / px_w
    metrics_mm["min_rect_h_mm"] = metrics_px["min_rect_h"] / px_h_rect

    # Área (px² → mm²)
    metrics_mm["area_mm2"] = metrics_px["area_px"] / (px_h * px_v)

    # Perímetro (usa média dos fatores como aproximação)
    metrics_mm["perimeter_mm"] = metrics_px["perimeter_px"] / px_avg

    # Elipse: decompõe os fatores horizontal/vertical com base no ângulo de rotação da elipse
    ellipse_angle_deg = metrics_px.get("ellipse_angle", 0.0)
    theta_el = np.radians(ellipse_angle_deg)

    # Fator de escala para a maior dimensão (major)
    denom_el_major = (np.cos(theta_el) / px_h) ** 2 + (np.sin(theta_el) / px_v) ** 2
    px_el_major = 1.0 / np.sqrt(denom_el_major) if denom_el_major > 0 else px_avg

    # Fator de escala para a menor dimensão (minor)
    denom_el_minor = (np.sin(theta_el) / px_h) ** 2 + (np.cos(theta_el) / px_v) ** 2
    px_el_minor = 1.0 / np.sqrt(denom_el_minor) if denom_el_minor > 0 else px_avg

    metrics_mm["ellipse_major_mm"] = metrics_px["ellipse_major_px"] / px_el_major
    metrics_mm["ellipse_minor_mm"] = metrics_px["ellipse_minor_px"] / px_el_minor

    # Seções transversais: converter de px para mm
    # As seções de "width" são perpendiculares ao eixo maior (orientação do min_rect_h),
    # e as de "length" são perpendiculares ao eixo menor (orientação do min_rect_w).
    for pos_pct in range(0, 101, 5):
        w_key = f"cross_width_{pos_pct}pct_px"
        if w_key in metrics_px and metrics_px[w_key] > 0:
            metrics_mm[f"cross_width_{pos_pct}pct_mm"] = metrics_px[w_key] / px_h_rect

        l_key = f"cross_length_{pos_pct}pct_px"
        if l_key in metrics_px and metrics_px[l_key] > 0:
            metrics_mm[f"cross_length_{pos_pct}pct_mm"] = metrics_px[l_key] / px_w

    if "corner_radius_px" in metrics_px:
        metrics_mm["corner_radius_mm"] = metrics_px["corner_radius_px"] / px_avg

    # Metadados de escala (para rastreabilidade)
    metrics_mm["px_per_mm_h"] = px_h
    metrics_mm["px_per_mm_v"] = px_v
    metrics_mm["anisotropy"] = scale["anisotropy"]
    metrics_mm["view_mode"] = scale["view_mode"]

    return metrics_mm

