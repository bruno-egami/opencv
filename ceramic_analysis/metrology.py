# -*- coding: utf-8 -*-
"""
Módulo de metrologia: calibração de escala via grade no MDF e conversão de medidas.

Responsável por:
1. Detectar as interseções da grade gravada a laser na base MDF
2. Calcular fatores de escala px_per_mm independentes para H e V
3. Verificar distorção anisotrópica e linearidade
4. Corrigir perspectiva via homografia (opcional)
5. Aplicar correção de paralaxe para vistas laterais
6. Converter todas as métricas de pixels para milímetros

Uso:
    from metrology import detect_grid, calibrate_scale, convert_measurements

    # Detectar grade na foto da base vazia
    grid_points = detect_grid(background_image)

    # Calcular escala
    scale = calibrate_scale(grid_points, view_mode="top")

    # Converter métricas
    metrics_mm = convert_measurements(metrics_px, scale)
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
# Detecção da grade no MDF
# ──────────────────────────────────────────────────────────────────────────────

def detect_grid(
    background_image: np.ndarray,
    min_intersections: int = 6
) -> np.ndarray:
    """
    Detecta as interseções da grade quadriculada gravada a laser no MDF.

    A grade consiste em linhas pretas (gravação a laser) sobre MDF claro,
    formando uma malha de 20×20mm. As interseções são detectadas via:
    1. Binarização para isolar as linhas escuras
    2. Detecção de linhas via HoughLinesP
    3. Cálculo dos pontos de interseção
    4. Refinamento subpixel com cornerSubPix

    Args:
        background_image: Foto da base MDF vazia (BGR, uint8).
        min_intersections: Mínimo de interseções necessárias para calibração.

    Returns:
        Array Nx2 com coordenadas (x, y) das interseções em pixels,
        ordenadas da esquerda-para-direita, cima-para-baixo.

    Raises:
        MetrologyError: Se poucas interseções forem detectadas.
    """
    logger.info("Detectando grade no MDF...")

    gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

    # Equalizar histograma para melhor contraste das linhas
    gray_eq = cv2.equalizeHist(gray)

    # Binarizar: as linhas do laser são ESCURAS sobre MDF CLARO
    # Threshold adaptativo funciona melhor que Otsu aqui porque
    # a iluminação pode não ser uniforme sobre toda a base
    binary = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15
    )

    # Refinamento morfológico: afinar linhas
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_thin)

    # Detectar linhas com Hough Probabilístico
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=20
    )

    if lines is None or len(lines) < 4:
        raise MetrologyError(
            "Poucas linhas detectadas na grade. Verifique:\n"
            "  - A gravação a laser está visível e com bom contraste?\n"
            "  - A imagem está focada?\n"
            "  - A iluminação é suficiente?"
        )

    # Separar linhas horizontais e verticais pelo ângulo
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

        if angle < 20:   # Quase horizontal (< 20° da horizontal)
            horizontal_lines.append(line[0])
        elif angle > 70:  # Quase vertical (> 70° da horizontal)
            vertical_lines.append(line[0])

    logger.debug(
        f"  Linhas detectadas: {len(horizontal_lines)} horizontais, "
        f"{len(vertical_lines)} verticais"
    )

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        raise MetrologyError(
            f"Insuficiente: {len(horizontal_lines)} linhas horizontais, "
            f"{len(vertical_lines)} verticais (mínimo 2 de cada)."
        )

    # Agrupar linhas próximas (mesma posição) via clustering por coordenada
    h_positions = _cluster_lines(horizontal_lines, axis="h")
    v_positions = _cluster_lines(vertical_lines, axis="v")

    logger.debug(
        f"  Linhas agrupadas: {len(h_positions)} horizontais, "
        f"{len(v_positions)} verticais"
    )

    # Calcular interseções
    intersections = []
    for h_line in h_positions:
        for v_line in v_positions:
            point = _line_intersection(h_line, v_line)
            if point is not None:
                # Verificar se o ponto está dentro da imagem
                h, w = gray.shape
                if 0 <= point[0] < w and 0 <= point[1] < h:
                    intersections.append(point)

    if len(intersections) < min_intersections:
        raise MetrologyError(
            f"Apenas {len(intersections)} interseções encontradas "
            f"(mínimo: {min_intersections}). Verifique a qualidade da grade."
        )

    intersections = np.array(intersections, dtype=np.float32)

    # Refinamento subpixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = intersections.reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    intersections = corners.reshape(-1, 2)

    # Ordenar: esquerda→direita, cima→baixo
    intersections = _sort_grid_points(intersections)

    logger.info(f"  ✓ {len(intersections)} interseções detectadas e refinadas")

    return intersections


def _cluster_lines(lines: list, axis: str, min_gap: int = 15) -> list:
    """
    Agrupa linhas próximas (representando a mesma linha da grade).

    Linhas são agrupadas pela coordenada perpendicular ao eixo:
    - Horizontais: agrupar por posição Y
    - Verticais: agrupar por posição X

    Args:
        lines: Lista de segmentos (x1, y1, x2, y2).
        axis: "h" para horizontais, "v" para verticais.
        min_gap: Distância mínima entre clusters (pixels).

    Returns:
        Lista de linhas representativas (uma por cluster), cada uma como
        tupla ((x1, y1), (x2, y2)) com os pontos médios do cluster.
    """
    if axis == "h":
        # Posição Y média de cada linha
        positions = [(y1 + y2) / 2 for x1, y1, x2, y2 in lines]
    else:
        # Posição X média de cada linha
        positions = [(x1 + x2) / 2 for x1, y1, x2, y2 in lines]

    # Ordenar por posição
    sorted_indices = np.argsort(positions)
    sorted_positions = [positions[i] for i in sorted_indices]
    sorted_lines = [lines[i] for i in sorted_indices]

    # Agrupar por proximidade
    clusters = []
    current_cluster = [sorted_lines[0]]
    current_pos = sorted_positions[0]

    for i in range(1, len(sorted_lines)):
        if sorted_positions[i] - current_pos < min_gap:
            current_cluster.append(sorted_lines[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_lines[i]]
        current_pos = sorted_positions[i]
    clusters.append(current_cluster)

    # Linha representativa: média dos endpoints do cluster
    representative_lines = []
    for cluster in clusters:
        x1_avg = np.mean([l[0] for l in cluster])
        y1_avg = np.mean([l[1] for l in cluster])
        x2_avg = np.mean([l[2] for l in cluster])
        y2_avg = np.mean([l[3] for l in cluster])
        representative_lines.append(
            ((x1_avg, y1_avg), (x2_avg, y2_avg))
        )

    return representative_lines


def _line_intersection(line1: tuple, line2: tuple):
    """
    Calcula o ponto de interseção entre duas linhas definidas por 2 pontos cada.

    Args:
        line1: Tupla ((x1, y1), (x2, y2)) da primeira linha.
        line2: Tupla ((x1, y1), (x2, y2)) da segunda linha.

    Returns:
        Tupla (x, y) do ponto de interseção, ou None se as linhas são paralelas.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # Linhas paralelas

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return (ix, iy)


def _sort_grid_points(points: np.ndarray) -> np.ndarray:
    """
    Ordena pontos de uma grade em ordem de leitura (esquerda→direita, cima→baixo).

    Usa clustering hierárquico na coordenada Y para identificar linhas,
    depois ordena cada linha por X.

    Args:
        points: Array Nx2 de coordenadas (x, y).

    Returns:
        Array Nx2 ordenado.
    """
    if len(points) < 2:
        return points

    # Ordenar por Y primeiro
    sorted_by_y = points[np.argsort(points[:, 1])]

    # Agrupar em linhas por proximidade em Y
    rows = []
    current_row = [sorted_by_y[0]]

    y_threshold = np.median(np.diff(np.sort(points[:, 1]))) * 0.5
    if y_threshold < 5:
        y_threshold = 15  # Fallback mínimo

    for i in range(1, len(sorted_by_y)):
        if abs(sorted_by_y[i, 1] - current_row[-1][1]) < y_threshold:
            current_row.append(sorted_by_y[i])
        else:
            rows.append(np.array(current_row))
            current_row = [sorted_by_y[i]]
    rows.append(np.array(current_row))

    # Ordenar cada linha por X
    sorted_points = []
    for row in rows:
        row_sorted = row[np.argsort(row[:, 0])]
        sorted_points.extend(row_sorted)

    return np.array(sorted_points)


# ──────────────────────────────────────────────────────────────────────────────
# Calibração de escala
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_scale(
    grid_points: np.ndarray,
    view_mode: str = "top",
    grid_spacing_mm: float = None,
    camera_matrix: np.ndarray = None
) -> dict:
    """
    Calcula fatores de escala px_per_mm a partir das interseções da grade.

    Calcula o espaçamento médio entre interseções adjacentes em ambos os
    eixos, dividindo pelo espaçamento real (20mm) para obter px_per_mm.

    Para vista lateral, aplica correção de paralaxe se configurada.

    Args:
        grid_points: Array Nx2 com interseções da grade em pixels.
        view_mode: "top" (vista de cima) ou "side" (vista lateral).
        grid_spacing_mm: Espaçamento real da grade em mm.
                         Default: config.GRID_SPACING_MM (20.0)
        camera_matrix: Matriz intrínseca (necessária para correção de paralaxe).
                       Default: None (usa aproximação geométrica).

    Returns:
        Dict com:
            - px_per_mm_h: Fator de escala horizontal (pixels por mm)
            - px_per_mm_v: Fator de escala vertical (pixels por mm)
            - anisotropy: Diferença relativa entre H e V (0.0 = isotrópico)
            - linearity_h: Desvio padrão relativo dos espaçamentos horizontais
            - linearity_v: Desvio padrão relativo dos espaçamentos verticais
            - n_points: Número de interseções usadas
            - view_mode: "top" ou "side"
    """
    grid_spacing_mm = grid_spacing_mm or config.GRID_SPACING_MM

    logger.info(f"Calibrando escala (vista: {view_mode})...")

    # Organizar pontos em linhas e colunas
    rows = _organize_into_rows(grid_points)

    # Calcular espaçamentos horizontais (entre colunas adjacentes na mesma linha)
    h_spacings = []
    for row in rows:
        if len(row) >= 2:
            sorted_row = row[np.argsort(row[:, 0])]
            diffs = np.diff(sorted_row[:, 0])
            h_spacings.extend(diffs)

    # Calcular espaçamentos verticais (entre linhas adjacentes na mesma coluna)
    cols = _organize_into_columns(grid_points)
    v_spacings = []
    for col in cols:
        if len(col) >= 2:
            sorted_col = col[np.argsort(col[:, 1])]
            diffs = np.diff(sorted_col[:, 1])
            v_spacings.extend(diffs)

    if not h_spacings or not v_spacings:
        raise MetrologyError(
            "Não foi possível calcular espaçamentos da grade. "
            "Verifique se há pelo menos 2×2 interseções detectadas."
        )

    h_spacings = np.array(h_spacings)
    v_spacings = np.array(v_spacings)

    # Filtrar outliers (espaçamentos muito diferentes da mediana)
    h_spacings = _filter_outliers(h_spacings)
    v_spacings = _filter_outliers(v_spacings)

    # Espaçamento médio em pixels
    mean_h_px = np.mean(h_spacings)
    mean_v_px = np.mean(v_spacings)

    # Fatores de escala
    px_per_mm_h = mean_h_px / grid_spacing_mm
    px_per_mm_v = mean_v_px / grid_spacing_mm

    # Aplicar correção de paralaxe para vista lateral
    if view_mode == "side" and config.SIDE_GRID_GAP_MM > 0:
        px_per_mm_h, px_per_mm_v = _apply_parallax_correction(
            px_per_mm_h, px_per_mm_v, camera_matrix, grid_points
        )

    # Verificar anisotropia
    anisotropy = abs(px_per_mm_h - px_per_mm_v) / max(px_per_mm_h, px_per_mm_v)

    # Verificar linearidade (distorção residual)
    linearity_h = np.std(h_spacings) / mean_h_px if mean_h_px > 0 else 0
    linearity_v = np.std(v_spacings) / mean_v_px if mean_v_px > 0 else 0

    # Logging
    logger.info(f"  Espaçamento horizontal: {mean_h_px:.2f} px = {grid_spacing_mm} mm")
    logger.info(f"  Espaçamento vertical:   {mean_v_px:.2f} px = {grid_spacing_mm} mm")
    logger.info(f"  px_per_mm_h: {px_per_mm_h:.4f}")
    logger.info(f"  px_per_mm_v: {px_per_mm_v:.4f}")

    if anisotropy > 0.02:
        logger.warning(
            f"  ⚠ Distorção anisotrópica detectada: {anisotropy:.1%}. "
            f"As escalas H e V diferem mais de 2%."
        )
    else:
        logger.info(f"  ✓ Anisotropia: {anisotropy:.1%} (< 2%)")

    if linearity_h > 0.02 or linearity_v > 0.02:
        logger.warning(
            f"  ⚠ Possível distorção radial residual: "
            f"linearidade H={linearity_h:.1%}, V={linearity_v:.1%}"
        )

    scale = {
        "px_per_mm_h": px_per_mm_h,
        "px_per_mm_v": px_per_mm_v,
        "anisotropy": anisotropy,
        "linearity_h": linearity_h,
        "linearity_v": linearity_v,
        "n_points": len(grid_points),
        "view_mode": view_mode,
    }

    return scale


def _apply_parallax_correction(
    px_per_mm_h: float,
    px_per_mm_v: float,
    camera_matrix: np.ndarray = None,
    grid_points: np.ndarray = None
) -> tuple:
    """
    Corrige a escala para a diferença de profundidade entre grade e peça.

    Na vista lateral, a grade está atrás da peça (mais longe da câmera).
    A escala medida na grade é ligeiramente menor que a escala real da peça.

    Correção: scale_peça = scale_grade × (dist_grade / dist_peça)
    Como dist_peça = dist_grade - gap:
        correction = dist_grade / (dist_grade - gap)

    A distância à grade é estimada a partir do focal length da câmera
    e do tamanho conhecido da grade. Se a câmera não foi calibrada,
    usa uma estimativa conservadora.

    Args:
        px_per_mm_h: Fator horizontal sem correção.
        px_per_mm_v: Fator vertical sem correção.
        camera_matrix: Matriz intrínseca (para focal length).
        grid_points: Pontos da grade (para estimar distância).

    Returns:
        Tupla (px_per_mm_h_corrigido, px_per_mm_v_corrigido).
    """
    gap_mm = config.SIDE_GRID_GAP_MM

    if gap_mm <= 0:
        return px_per_mm_h, px_per_mm_v

    # Estimar distância câmera→grade via focal length e tamanho da grade
    if camera_matrix is not None and grid_points is not None:
        focal_px = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2

        # Tamanho da grade na imagem (pixels)
        grid_width_px = np.max(grid_points[:, 0]) - np.min(grid_points[:, 0])
        grid_width_mm = config.MDF_WIDTH_MM

        # Distância = focal × (tamanho_real / tamanho_imagem)
        dist_grade_mm = focal_px * (grid_width_mm / grid_width_px)
    else:
        # Estimativa conservadora: câmera a ~600mm da grade
        dist_grade_mm = 600.0
        logger.debug(
            f"  Sem calibração — estimando distância câmera→grade: {dist_grade_mm}mm"
        )

    dist_peca_mm = dist_grade_mm - gap_mm
    correction = dist_grade_mm / dist_peca_mm

    logger.info(
        f"  Correção de paralaxe: gap={gap_mm}mm, "
        f"dist_grade≈{dist_grade_mm:.0f}mm, "
        f"fator={correction:.4f} ({(correction - 1) * 100:.2f}%)"
    )

    return px_per_mm_h * correction, px_per_mm_v * correction


def _organize_into_rows(points: np.ndarray, y_threshold: float = None) -> list:
    """Organiza pontos em linhas horizontais por proximidade em Y."""
    if len(points) < 2:
        return [points]

    sorted_by_y = points[np.argsort(points[:, 1])]
    y_diffs = np.diff(sorted_by_y[:, 1])

    if y_threshold is None:
        # Usar mediana dos diffs como threshold
        y_threshold = np.median(y_diffs[y_diffs > 5]) * 0.5 if len(y_diffs[y_diffs > 5]) > 0 else 15

    rows = []
    current_row = [sorted_by_y[0]]

    for i in range(1, len(sorted_by_y)):
        if abs(sorted_by_y[i, 1] - current_row[-1][1]) < y_threshold:
            current_row.append(sorted_by_y[i])
        else:
            rows.append(np.array(current_row))
            current_row = [sorted_by_y[i]]
    rows.append(np.array(current_row))

    return rows


def _organize_into_columns(points: np.ndarray, x_threshold: float = None) -> list:
    """Organiza pontos em colunas verticais por proximidade em X."""
    if len(points) < 2:
        return [points]

    sorted_by_x = points[np.argsort(points[:, 0])]
    x_diffs = np.diff(sorted_by_x[:, 0])

    if x_threshold is None:
        x_threshold = np.median(x_diffs[x_diffs > 5]) * 0.5 if len(x_diffs[x_diffs > 5]) > 0 else 15

    cols = []
    current_col = [sorted_by_x[0]]

    for i in range(1, len(sorted_by_x)):
        if abs(sorted_by_x[i, 0] - current_col[-1][0]) < x_threshold:
            current_col.append(sorted_by_x[i])
        else:
            cols.append(np.array(current_col))
            current_col = [sorted_by_x[i]]
    cols.append(np.array(current_col))

    return cols


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


# ──────────────────────────────────────────────────────────────────────────────
# Correção de perspectiva
# ──────────────────────────────────────────────────────────────────────────────

def correct_perspective(
    image: np.ndarray,
    detected_points: np.ndarray,
    grid_spacing_mm: float = None,
    output_px_per_mm: float = None
) -> tuple:
    """
    Corrige distorção de perspectiva usando pontos da grade como referência.

    Se a câmera não está perfeitamente perpendicular à base/backdrop,
    a homografia transforma a imagem para que a grade fique retangular.

    Args:
        image: Imagem para corrigir (BGR ou gray).
        detected_points: Pontos detectados da grade (Nx2).
        grid_spacing_mm: Espaçamento real da grade.
        output_px_per_mm: Se definido, escala a imagem para esta resolução.

    Returns:
        Tupla (corrected_image, H):
            - corrected_image: Imagem corrigida.
            - H: Matriz de homografia 3x3.
    """
    grid_spacing_mm = grid_spacing_mm or config.GRID_SPACING_MM

    # Organizar em grade
    rows = _organize_into_rows(detected_points)
    n_rows = len(rows)
    n_cols = min(len(r) for r in rows) if rows else 0

    if n_rows < 2 or n_cols < 2:
        raise MetrologyError("Grade insuficiente para correção de perspectiva")

    # Criar pontos ideais (grade perfeita)
    if output_px_per_mm is None:
        # Manter resolução aproximada da imagem original
        mean_spacing = np.mean([
            np.mean(np.diff(np.sort(row[:n_cols, 0])))
            for row in rows if len(row) >= n_cols
        ])
        output_px_per_mm = mean_spacing / grid_spacing_mm

    src_points = []
    dst_points = []

    for i, row in enumerate(rows):
        sorted_row = row[np.argsort(row[:, 0])]
        for j in range(min(n_cols, len(sorted_row))):
            src_points.append(sorted_row[j])
            # Posição ideal: grade perfeita com espaçamento uniforme
            ideal_x = j * grid_spacing_mm * output_px_per_mm + 50
            ideal_y = i * grid_spacing_mm * output_px_per_mm + 50
            dst_points.append([ideal_x, ideal_y])

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Calcular homografia
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3.0)

    if H is None:
        raise MetrologyError("Falha ao calcular homografia para correção de perspectiva")

    # Aplicar transformação
    h, w = image.shape[:2]
    # Calcular tamanho da imagem de saída
    out_w = int(n_cols * grid_spacing_mm * output_px_per_mm + 100)
    out_h = int(n_rows * grid_spacing_mm * output_px_per_mm + 100)
    out_w = max(out_w, w)
    out_h = max(out_h, h)

    corrected = cv2.warpPerspective(image, H, (out_w, out_h))

    logger.info(
        f"  ✓ Perspectiva corrigida: {w}×{h} → {out_w}×{out_h}, "
        f"px_per_mm={output_px_per_mm:.2f}"
    )

    return corrected, H


# ──────────────────────────────────────────────────────────────────────────────
# Conversão de medidas px → mm
# ──────────────────────────────────────────────────────────────────────────────

def convert_measurements(metrics_px: dict, scale: dict) -> dict:
    """
    Converte métricas de pixels para milímetros usando os fatores de escala.

    Utiliza fatores independentes para H e V para lidar com anisotropia.

    Args:
        metrics_px: Dict com métricas em pixels (de segmentation.extract_contour_metrics).
        scale: Dict com fatores de escala (de calibrate_scale).

    Returns:
        Dict com métricas originais + métricas em mm adicionadas.
    """
    px_h = scale["px_per_mm_h"]
    px_v = scale["px_per_mm_v"]

    metrics_mm = dict(metrics_px)  # Copiar originais

    # Bounding box
    metrics_mm["bbox_w_mm"] = metrics_px["bbox_w"] / px_h
    metrics_mm["bbox_h_mm"] = metrics_px["bbox_h"] / px_v

    # Retângulo mínimo
    # Para o minAreaRect, a orientação depende do ângulo.
    # Como aproximação, usamos a média dos fatores para as dimensões
    # do retângulo rotacionado (a decomposição exata requer o ângulo).
    px_avg = (px_h + px_v) / 2
    metrics_mm["min_rect_w_mm"] = metrics_px["min_rect_w"] / px_avg
    metrics_mm["min_rect_h_mm"] = metrics_px["min_rect_h"] / px_avg

    # Área (px² → mm²)
    metrics_mm["area_mm2"] = metrics_px["area_px"] / (px_h * px_v)

    # Perímetro (usa média dos fatores como aproximação)
    metrics_mm["perimeter_mm"] = metrics_px["perimeter_px"] / px_avg

    # Elipse
    metrics_mm["ellipse_major_mm"] = metrics_px["ellipse_major_px"] / px_avg
    metrics_mm["ellipse_minor_mm"] = metrics_px["ellipse_minor_px"] / px_avg

    # Metadados de escala (para rastreabilidade)
    metrics_mm["px_per_mm_h"] = px_h
    metrics_mm["px_per_mm_v"] = px_v
    metrics_mm["anisotropy"] = scale["anisotropy"]
    metrics_mm["view_mode"] = scale["view_mode"]

    return metrics_mm


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: seleção manual
# ──────────────────────────────────────────────────────────────────────────────

def manual_scale_calibration(image: np.ndarray, view_mode: str = "top") -> dict:
    """
    Calibração manual de escala via seleção interativa de ROI.

    O usuário seleciona dois pontos com distância conhecida na imagem,
    uma vez na horizontal e uma vez na vertical.

    Args:
        image: Imagem para calibração (BGR).
        view_mode: "top" ou "side".

    Returns:
        Dict com fatores de escala (mesmo formato de calibrate_scale).
    """
    logger.info("Calibração manual de escala")
    logger.info("  Selecione um segmento HORIZONTAL de comprimento conhecido...")
    logger.info("  (Pressione ENTER para confirmar, ESC para cancelar)")

    # Seleção horizontal
    roi_h = cv2.selectROI("Selecione segmento horizontal", image, False)
    cv2.destroyWindow("Selecione segmento horizontal")

    if roi_h[2] == 0:
        raise MetrologyError("Calibração manual cancelada pelo usuário")

    dist_h_mm = float(input("Comprimento real do segmento horizontal (mm): "))
    px_per_mm_h = roi_h[2] / dist_h_mm

    # Seleção vertical
    logger.info("  Selecione um segmento VERTICAL de comprimento conhecido...")
    roi_v = cv2.selectROI("Selecione segmento vertical", image, False)
    cv2.destroyWindow("Selecione segmento vertical")

    if roi_v[3] == 0:
        raise MetrologyError("Calibração manual cancelada pelo usuário")

    dist_v_mm = float(input("Comprimento real do segmento vertical (mm): "))
    px_per_mm_v = roi_v[3] / dist_v_mm

    anisotropy = abs(px_per_mm_h - px_per_mm_v) / max(px_per_mm_h, px_per_mm_v)

    logger.info(f"  ✓ Manual: px_per_mm_h={px_per_mm_h:.4f}, "
                f"px_per_mm_v={px_per_mm_v:.4f}")

    return {
        "px_per_mm_h": px_per_mm_h,
        "px_per_mm_v": px_per_mm_v,
        "anisotropy": anisotropy,
        "linearity_h": 0.0,
        "linearity_v": 0.0,
        "n_points": 2,
        "view_mode": view_mode,
    }
