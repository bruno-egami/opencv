# -*- coding: utf-8 -*-
"""
Módulo de análise comparativa e exportação de dados.

Responsável por:
1. Correlacionar medições de peças úmidas e secas
2. Combinar vistas (top + side) para dimensões 3D
3. Calcular retração percentual: ΔL(%) = (Lúmido - Lseco) / Lúmido × 100
4. Exportar resultados para CSV
5. Gerar imagens anotadas com contornos e dimensões

Uso:
    from analysis import compare_specimens, export_csv, annotate_image

    shrinkage = compare_specimens(wet_results, dry_results)
    export_csv(all_results, "output/results.csv")
    annotate_image(image, metrics_mm, "output/annotated/sample_01.png")
"""

import csv
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Retração percentual
# ──────────────────────────────────────────────────────────────────────────────

def calculate_shrinkage(wet_value: float, dry_value: float) -> float:
    """
    Calcula a retração percentual entre o estado úmido e seco.

    Fórmula: ΔL(%) = (L_úmido - L_seco) / L_úmido × 100

    Args:
        wet_value: Medida no estado úmido (mm).
        dry_value: Medida no estado seco (mm).

    Returns:
        Retração percentual (%). Valores positivos indicam encolhimento.
        Retorna 0.0 se wet_value é zero (evita divisão por zero).
    """
    if wet_value == 0:
        logger.warning("Valor úmido é zero — retração não calculável")
        return 0.0

    return ((wet_value - dry_value) / wet_value) * 100.0


def compare_specimens(
    wet_results: list,
    dry_results: list,
    match_by: str = "sample_id"
) -> list:
    """
    Compara medições de peças úmidas e secas e calcula retração.

    Correlaciona peças pelo sample_id (derivado do nome do arquivo).

    Args:
        wet_results: Lista de dicts com métricas em mm das peças úmidas.
        dry_results: Lista de dicts com métricas em mm das peças secas.
        match_by: Campo usado para correlacionar (default: "sample_id").

    Returns:
        Lista de dicts com dados de ambos os estados e retração calculada.
    """
    logger.info(
        f"Comparando {len(wet_results)} peça(s) úmida(s) com "
        f"{len(dry_results)} peça(s) seca(s)..."
    )

    # Indexar por sample_id
    wet_by_id = {}
    for r in wet_results:
        sid = r.get(match_by, r.get("image_name", "unknown"))
        wet_by_id[sid] = r

    dry_by_id = {}
    for r in dry_results:
        sid = r.get(match_by, r.get("image_name", "unknown"))
        dry_by_id[sid] = r

    # Encontrar pares
    common_ids = set(wet_by_id.keys()) & set(dry_by_id.keys())

    if not common_ids:
        logger.warning(
            "Nenhum par úmido-seco encontrado! Verifique se os nomes dos "
            "arquivos correspondem entre wet/ e dry/."
        )
        # Retornar dados individuais sem retração
        all_results = []
        for r in wet_results:
            r["state"] = "wet"
            all_results.append(r)
        for r in dry_results:
            r["state"] = "dry"
            all_results.append(r)
        return all_results

    logger.info(f"  {len(common_ids)} par(es) encontrado(s): {sorted(common_ids)}")

    results = []
    for sid in sorted(common_ids):
        wet = wet_by_id[sid]
        dry = dry_by_id[sid]

        comparison = {
            "sample_id": sid,
            "session": wet.get("session", ""),
            "view_mode": wet.get("view_mode", "top"),
        }

        # Dimensões para calcular retração
        dimension_pairs = [
            ("bbox_w_mm", "shrinkage_width_pct"),
            ("bbox_h_mm", "shrinkage_height_pct"),
            ("area_mm2", "shrinkage_area_pct"),
            ("perimeter_mm", "shrinkage_perimeter_pct"),
            ("ellipse_major_mm", "shrinkage_ellipse_major_pct"),
            ("ellipse_minor_mm", "shrinkage_ellipse_minor_pct"),
            ("min_rect_w_mm", "shrinkage_min_rect_w_pct"),
            ("min_rect_h_mm", "shrinkage_min_rect_h_pct"),
        ]

        for dim_key, shrink_key in dimension_pairs:
            wet_val = wet.get(dim_key, 0.0)
            dry_val = dry.get(dim_key, 0.0)

            comparison[f"wet_{dim_key}"] = wet_val
            comparison[f"dry_{dim_key}"] = dry_val
            comparison[shrink_key] = calculate_shrinkage(wet_val, dry_val)

        # Metadados de escala
        comparison["wet_px_per_mm_h"] = wet.get("px_per_mm_h", 0)
        comparison["wet_px_per_mm_v"] = wet.get("px_per_mm_v", 0)
        comparison["dry_px_per_mm_h"] = dry.get("px_per_mm_h", 0)
        comparison["dry_px_per_mm_v"] = dry.get("px_per_mm_v", 0)
        comparison["wet_anisotropy"] = wet.get("anisotropy", 0)
        comparison["dry_anisotropy"] = dry.get("anisotropy", 0)

        # Circularidade (não tem retração, mas mostra mudança de forma)
        comparison["wet_circularity"] = wet.get("circularity", 0)
        comparison["dry_circularity"] = dry.get("circularity", 0)

        results.append(comparison)

        # Log resumido
        sw = comparison.get("shrinkage_width_pct", 0)
        sh = comparison.get("shrinkage_height_pct", 0)
        sa = comparison.get("shrinkage_area_pct", 0)
        logger.info(
            f"  {sid}: ΔLargura={sw:.2f}%, ΔAltura={sh:.2f}%, ΔÁrea={sa:.2f}%"
        )

    # Adicionar peças sem par
    unmatched_wet = set(wet_by_id.keys()) - common_ids
    unmatched_dry = set(dry_by_id.keys()) - common_ids

    if unmatched_wet:
        logger.warning(f"  Peças úmidas sem par seco: {sorted(unmatched_wet)}")
    if unmatched_dry:
        logger.warning(f"  Peças secas sem par úmido: {sorted(unmatched_dry)}")

    return results


def combine_views(top_results: list, side_results: list) -> list:
    """
    Combina medições das vistas top e side para dimensões 3D.

    Vista top → Largura (X), Profundidade (Y)
    Vista side → Largura (X, verificação cruzada), Altura (Z)

    Args:
        top_results: Lista de medições da vista de cima.
        side_results: Lista de medições da vista lateral.

    Returns:
        Lista de medições combinadas com dimensões 3D.
    """
    logger.info("Combinando vistas top + side para dimensões 3D...")

    # Indexar por sample_id
    top_by_id = {r.get("sample_id", r.get("image_name")): r for r in top_results}
    side_by_id = {r.get("sample_id", r.get("image_name")): r for r in side_results}

    common_ids = set(top_by_id.keys()) & set(side_by_id.keys())
    combined = []

    for sid in sorted(common_ids):
        top = top_by_id[sid]
        side = side_by_id[sid]

        result = dict(top)  # Base: dados do top
        result["sample_id"] = sid

        # Dimensões 3D
        # Top: bbox_w_mm = Largura (X), bbox_h_mm = Profundidade (Y)
        # Side: bbox_w_mm = Largura (X), bbox_h_mm = Altura (Z)
        result["width_mm"] = top.get("bbox_w_mm", 0)       # X (do top)
        result["depth_mm"] = top.get("bbox_h_mm", 0)       # Y (do top)
        result["height_mm"] = side.get("bbox_h_mm", 0)     # Z (do side)

        # Cross-validation: largura deve ser similar em ambas as vistas
        width_top = top.get("bbox_w_mm", 0)
        width_side = side.get("bbox_w_mm", 0)

        if width_top > 0 and width_side > 0:
            width_diff = abs(width_top - width_side) / max(width_top, width_side)
            result["width_cross_validation_pct"] = width_diff * 100

            if width_diff > 0.05:
                logger.warning(
                    f"  ⚠ {sid}: Largura difere entre vistas: "
                    f"top={width_top:.2f}mm, side={width_side:.2f}mm "
                    f"(diferença: {width_diff:.1%})"
                )
            else:
                logger.info(
                    f"  ✓ {sid}: Largura consistente: top={width_top:.2f}mm, "
                    f"side={width_side:.2f}mm ({width_diff:.1%})"
                )

        # Volume aproximado (se geometria simples)
        if result["width_mm"] > 0 and result["depth_mm"] > 0 and result["height_mm"] > 0:
            # Para prisma retangular
            result["volume_mm3"] = (
                result["width_mm"] * result["depth_mm"] * result["height_mm"]
            )

        combined.append(result)

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Exportação CSV
# ──────────────────────────────────────────────────────────────────────────────

# Colunas do CSV em ordem
CSV_COLUMNS = [
    "sample_id", "session", "view_mode", "state",
    "bbox_w_mm", "bbox_h_mm", "area_mm2", "perimeter_mm",
    "min_rect_w_mm", "min_rect_h_mm",
    "ellipse_major_mm", "ellipse_minor_mm",
    "circularity", "solidity",
    "px_per_mm_h", "px_per_mm_v", "anisotropy",
    "width_mm", "depth_mm", "height_mm", "volume_mm3",
    "width_cross_validation_pct",
]

CSV_SHRINKAGE_COLUMNS = [
    "sample_id", "session", "view_mode",
    "wet_bbox_w_mm", "dry_bbox_w_mm", "shrinkage_width_pct",
    "wet_bbox_h_mm", "dry_bbox_h_mm", "shrinkage_height_pct",
    "wet_area_mm2", "dry_area_mm2", "shrinkage_area_pct",
    "wet_perimeter_mm", "dry_perimeter_mm", "shrinkage_perimeter_pct",
    "wet_ellipse_major_mm", "dry_ellipse_major_mm", "shrinkage_ellipse_major_pct",
    "wet_ellipse_minor_mm", "dry_ellipse_minor_mm", "shrinkage_ellipse_minor_pct",
    "wet_circularity", "dry_circularity",
    "wet_px_per_mm_h", "wet_px_per_mm_v", "wet_anisotropy",
    "dry_px_per_mm_h", "dry_px_per_mm_v", "dry_anisotropy",
]

CSV_CAD_COLUMNS = [
    "sample_id", "session", "cad_view", "photo_view", "state",
    "cad_model", "shape_class", "shape_complexity",
    # Dimensões CAD
    "cad_bbox_w_mm", "cad_bbox_h_mm", "cad_area_mm2",
    # Dimensões medidas
    "measured_bbox_w_mm", "measured_bbox_h_mm", "measured_area_mm2",
    # Desvios dimensionais
    "bbox_w_deviation_mm", "bbox_h_deviation_mm",
    "bbox_w_deviation_pct", "bbox_h_deviation_pct",
    "area_deviation_pct",
    # Desvios geométricos (contorno)
    "hausdorff_mm", "mean_deviation_mm", "deviation_std_mm", "deviation_p95_mm",
    "iou",
    # Diâmetro (axissimétricos)
    "diameter_cad_mm", "diameter_photo_mm",
    "diameter_deviation_mm", "diameter_deviation_pct",
    "concentricity_mm",
    # Furos (orgânicas)
    "n_holes_cad", "n_holes_photo", "holes_matched", "holes_iou",
    # Registro
    "registration_method", "registration_rotation_deg", "registration_rms_mm",
    # Orientação
    "auto_oriented", "cad_extents_mm",
]


def export_csv(data: list, output_path: str = None, columns: list = None):
    """
    Exporta dados de medição para arquivo CSV.

    Args:
        data: Lista de dicts com dados de medição.
        output_path: Caminho do CSV de saída. Default: config.RESULTS_CSV.
        columns: Lista de colunas a incluir. Default: auto-detecta.
    """
    output_path = Path(output_path or config.RESULTS_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        logger.warning("Nenhum dado para exportar")
        return

    # Auto-detectar colunas se não especificadas
    if columns is None:
        # Verificar se são dados de retração, comparação CAD ou medição individual
        if "shrinkage_width_pct" in data[0]:
            columns = CSV_SHRINKAGE_COLUMNS
        elif "hausdorff_mm" in data[0]:
            columns = CSV_CAD_COLUMNS
        else:
            columns = CSV_COLUMNS


    # Filtrar colunas que existem nos dados
    available_columns = [c for c in columns if any(c in d for d in data)]

    # Adicionar colunas extras presentes nos dados mas não na lista
    for d in data:
        for key in d:
            if key not in available_columns and not key.startswith("contour") and key != "hull":
                if isinstance(d[key], (int, float, str, bool, type(None))):
                    available_columns.append(key)
    # Remove duplicates preserving order
    seen = set()
    unique_columns = []
    for c in available_columns:
        if c not in seen:
            seen.add(c)
            unique_columns.append(c)
    available_columns = unique_columns

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=available_columns, extrasaction="ignore"
        )
        writer.writeheader()

        for row in data:
            # Filtrar campos não serializáveis (contornos numpy)
            clean_row = {}
            for key in available_columns:
                value = row.get(key, "")
                if isinstance(value, (np.floating, np.integer)):
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    continue
                clean_row[key] = value
            writer.writerow(clean_row)

    logger.info(f"  ✓ CSV exportado: {output_path} ({len(data)} linhas)")


# ──────────────────────────────────────────────────────────────────────────────
# Anotação de imagens
# ──────────────────────────────────────────────────────────────────────────────

def annotate_image(
    image: np.ndarray,
    metrics: dict,
    output_path: str,
    scale: dict = None,
    draw_contour: bool = True,
    draw_bbox: bool = True,
    draw_ellipse: bool = False,
    draw_dimensions: bool = True,
    dim_background: bool = True,
) -> np.ndarray:
    """
    Desenha contornos, dimensões e metadados sobre a imagem.

    Args:
        image: Imagem BGR para anotar (será copiada).
        metrics: Dict com métricas (px e mm) do contorno.
        output_path: Caminho para salvar a imagem anotada.
        scale: Dict com informações de escala (para metadados no canto).
        draw_contour: Desenhar o contorno detectado.
        draw_bbox: Desenhar o bounding box.
        draw_ellipse: Desenhar a elipse ajustada.
        draw_dimensions: Sobrepor dimensões em mm.
        dim_background: Escurecer o fundo (fora do contorno) para destacar a peça.

    Returns:
        Imagem anotada.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Escurecer o background (fora do contorno da peça) em 50% para destacá-la
    if dim_background and "contour" in metrics:
        contour = metrics["contour"]
        mask_inside = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_inside, [contour], -1, 255, -1)
        mask_outside = cv2.bitwise_not(mask_inside)
        
        # Gerar background escurecido
        darkened_bg = cv2.addWeighted(annotated, 0.5, np.zeros_like(annotated), 0.5, 0)
        
        # Combinar: área interna mantém original, área externa fica escurecida
        img_inside = cv2.bitwise_and(annotated, annotated, mask=mask_inside)
        img_outside = cv2.bitwise_and(darkened_bg, darkened_bg, mask=mask_outside)
        annotated = cv2.add(img_inside, img_outside)

    # Cores
    COLOR_CONTOUR = (0, 255, 0)      # Verde
    COLOR_BBOX = (255, 200, 0)       # Ciano
    COLOR_ELLIPSE = (0, 165, 255)    # Laranja
    COLOR_TEXT = (255, 255, 255)      # Branco
    COLOR_TEXT_BG = (0, 0, 0)        # Preto (fundo do texto)

    # Escala da fonte relativa ao tamanho da imagem
    font_scale = max(0.5, min(w, h) / 1500.0)
    thickness = max(1, int(font_scale * 2))

    # Contorno
    if draw_contour and "contour" in metrics:
        cv2.drawContours(annotated, [metrics["contour"]], -1, COLOR_CONTOUR, 2)

    # Bounding box Rotacionado (Substitui o BBox upright)
    if draw_bbox and "min_rect_w" in metrics:
        cx = metrics.get("robust_center_x", metrics["min_rect_center_x"])
        cy = metrics.get("robust_center_y", metrics["min_rect_center_y"])
        center = (cx, cy)
        w = metrics.get("robust_w", metrics["min_rect_w"])
        h = metrics.get("robust_h", metrics["min_rect_h"])
        size = (w, h)
        angle = metrics["min_rect_angle"]
        box = cv2.boxPoints((center, size, angle))
        box = np.int32(box)
        cv2.drawContours(annotated, [box], 0, COLOR_BBOX, 2)
        
        # Opcional: Desenhar os ângulos internos se disponíveis
        if "corner_angle_0" in metrics and metrics.get("corner_angle_0", 0) > 0:
            corners = metrics.get("corners_px", box)
            if corners is not None and len(corners) == 4:
                for i in range(4):
                    pt = tuple(int(x) for x in corners[i])
                    angle_val = metrics.get(f"corner_angle_{i}", 90.0)
                    cv2.putText(annotated, f"{angle_val:.1f}o", 
                               (pt[0] - 20, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale * 0.8, COLOR_BBOX, thickness)

    # Elipse
    if draw_ellipse and metrics.get("ellipse_major_px", 0) > 0:
        try:
            center = (
                int(metrics.get("min_rect_center_x", w // 2)),
                int(metrics.get("min_rect_center_y", h // 2))
            )
            axes = (
                int(metrics["ellipse_major_px"] / 2),
                int(metrics["ellipse_minor_px"] / 2)
            )
            angle = metrics.get("ellipse_angle", 0)
            cv2.ellipse(annotated, center, axes, angle, 0, 360, COLOR_ELLIPSE, 2)
        except (ValueError, cv2.error):
            pass

    # Dimensões em mm
    if draw_dimensions:
        texts = []

        if "area_mm2" in metrics:
            texts.append(f"Area: {metrics['area_mm2']:.2f} mm²")
        if "perimeter_mm" in metrics:
            texts.append(f"Perim: {metrics['perimeter_mm']:.2f} mm")
        if "circularity" in metrics:
            texts.append(f"Circ: {metrics['circularity']:.3f}")

        # Posicionar texto no canto superior esquerdo (calculado dinamicamente para não cortar)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw_temp, th_temp), baseline_temp = cv2.getTextSize("Ag", font, font_scale, thickness)
        
        # Margens horizontais e verticais escalonadas com base na fonte para evitar cortes nas bordas
        x_start = int(25 * font_scale)
        y_offset = int(th_temp + 25 * font_scale)
        
        for text in texts:
            _draw_text_with_bg(
                annotated, text, (x_start, y_offset),
                font_scale, COLOR_TEXT, COLOR_TEXT_BG, thickness
            )
            y_offset += int(35 * font_scale) + int(10 * font_scale)

        # Desenhar cotas/linhas de dimensão diretamente sobre a peça no perímetro
        circularity = metrics.get("circularity", 0.0)
        
        if circularity <= 0.80 and "min_rect_center_x" in metrics and "min_rect_w" in metrics and "min_rect_h" in metrics:
            center_pt = (metrics["min_rect_center_x"], metrics["min_rect_center_y"])
            w_px = metrics["min_rect_w"]
            h_px = metrics["min_rect_h"]
            angle = metrics["min_rect_angle"]
            
            # Obter os 4 vértices do min_rect
            box = cv2.boxPoints((center_pt, (w_px, h_px), angle))
            
            # Fatores de escala
            px_h = scale.get("px_per_mm_h", 1.0) if scale else 1.0
            px_v = scale.get("px_per_mm_v", 1.0) if scale else 1.0
            
            # Desenhar cotas para os 4 segmentos do retângulo orientado
            # Descobrir qual é major e minor do min_rect
            rect_w_mm = metrics.get("min_rect_w_mm", 0)
            rect_h_mm = metrics.get("min_rect_h_mm", 0)
            major_mm = max(rect_w_mm, rect_h_mm)
            minor_mm = min(rect_w_mm, rect_h_mm)

            drawn_length = False
            drawn_width = False

            for i in range(4):
                p1 = box[i]
                p2 = box[(i + 1) % 4]
                
                # Calcular comprimento em mm deste segmento
                dx_mm = (p2[0] - p1[0]) / px_h
                dy_mm = (p2[1] - p1[1]) / px_v
                len_mm = np.sqrt(dx_mm**2 + dy_mm**2)
                
                # Evitar cotar segmentos excessivamente pequenos
                if len_mm < 1.0:
                    continue
                    
                if abs(len_mm - major_mm) <= abs(len_mm - minor_mm):
                    # Edge de comprimento (Length)
                    if not drawn_length:
                        if "cross_length_10pct_mm" in metrics:
                            texts = [
                                f"{metrics['cross_length_10pct_mm']:.1f} mm",
                                f"{metrics['cross_length_50pct_mm']:.1f} mm",
                                f"{metrics['cross_length_90pct_mm']:.1f} mm"
                            ]
                            _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, custom_texts=texts, offset_multiplier=2.0)
                        else:
                            _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, offset_multiplier=2.0)
                        drawn_length = True
                else:
                    # Edge de largura (Width)
                    if not drawn_width:
                        if "cross_width_10pct_mm" in metrics:
                            texts = [
                                f"{metrics['cross_width_10pct_mm']:.1f} mm",
                                f"{metrics['cross_width_50pct_mm']:.1f} mm",
                                f"{metrics['cross_width_90pct_mm']:.1f} mm"
                            ]
                            _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, custom_texts=texts, offset_multiplier=2.0)
                        else:
                            _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, offset_multiplier=2.0)
                        drawn_width = True
        elif "bbox_x" in metrics and "bbox_y" in metrics and "bbox_w" in metrics and "bbox_h" in metrics:
            x, y = metrics["bbox_x"], metrics["bbox_y"]
            bw, bh = metrics["bbox_w"], metrics["bbox_h"]
            
            offset = int(40 * font_scale)
            tick_size = int(6 * font_scale)
            
            # Cota Horizontal (Largura)
            if "bbox_w_mm" in metrics:
                # Determinar se desenha acima ou abaixo do bounding box
                if y - offset - 10 < 0:
                    dy = bh + offset  # Desenhar abaixo
                else:
                    dy = -offset      # Desenhar acima
                
                cota_y = y + dy
                # Linhas de extensão
                cv2.line(annotated, (x, y), (x, cota_y + (5 if dy < 0 else -5)), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x + bw, y), (x + bw, cota_y + (5 if dy < 0 else -5)), COLOR_BBOX, 1, cv2.LINE_AA)
                # Linha de cota principal
                cv2.line(annotated, (x, cota_y), (x + bw, cota_y), COLOR_BBOX, 1, cv2.LINE_AA)
                # Ticks arquitetônicos (traços inclinados a 45 graus)
                cv2.line(annotated, (x - tick_size, cota_y + tick_size), (x + tick_size, cota_y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                cv2.line(annotated, (x + bw - tick_size, cota_y + tick_size), (x + bw + tick_size, cota_y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                # Texto da cota horizontal
                text_w = f"{metrics['bbox_w_mm']:.2f} mm"
                _draw_text_with_bg(
                    annotated, text_w, (x + bw // 2, cota_y),
                    font_scale * 0.7, COLOR_TEXT, COLOR_TEXT_BG, max(1, thickness - 1),
                    center=True
                )
                
            # Cota Vertical (Altura)
            if "bbox_h_mm" in metrics:
                # Determinar se desenha à esquerda ou à direita
                if x - offset - 10 < 0:
                    dx = bw + offset  # Desenhar à direita
                else:
                    dx = -offset      # Desenhar à esquerda
                    
                cota_x = x + dx
                # Linhas de extensão
                cv2.line(annotated, (x, y), (cota_x + (5 if dx < 0 else -5), y), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x, y + bh), (cota_x + (5 if dx < 0 else -5), y + bh), COLOR_BBOX, 1, cv2.LINE_AA)
                # Linha de cota principal
                cv2.line(annotated, (cota_x, y), (cota_x, y + bh), COLOR_BBOX, 1, cv2.LINE_AA)
                # Ticks arquitetônicos
                cv2.line(annotated, (cota_x - tick_size, y + tick_size), (cota_x + tick_size, y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                cv2.line(annotated, (cota_x - tick_size, y + bh + tick_size), (cota_x + tick_size, y + bh - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                # Texto da cota vertical
                text_h = f"{metrics['bbox_h_mm']:.2f} mm"
                _draw_text_with_bg(
                    annotated, text_h, (cota_x, y + bh // 2),
                    font_scale * 0.7, COLOR_TEXT, COLOR_TEXT_BG, max(1, thickness - 1),
                    center=True
                )



    # Metadados de escala no canto inferior esquerdo
    if scale:
        meta_texts = [
            f"Vista: {scale.get('view_mode', 'N/A')}",
            f"px/mm H: {scale.get('px_per_mm_h', 0):.2f}",
            f"px/mm V: {scale.get('px_per_mm_v', 0):.2f}",
        ]
        x_start = int(25 * font_scale)
        y_offset_meta = h - int(25 * font_scale)
        for text in reversed(meta_texts):
            _draw_text_with_bg(
                annotated, text, (x_start, y_offset_meta),
                font_scale * 0.7, (200, 200, 200), COLOR_TEXT_BG, max(1, thickness - 1)
            )
            y_offset_meta -= int(25 * font_scale) + int(5 * font_scale)

    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.debug(f"  Imagem anotada salva: {output_path}")

    return annotated


def _draw_text_with_bg(
    img, text, position, font_scale, color, bg_color, thickness, center=False, angle=0
):
    """Desenha texto com fundo semi-transparente para legibilidade."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = 4
    pad_y_top = 5
    pad_y_bottom = baseline + 2
    
    box_w = tw + 2 * pad_x
    box_h = th + pad_y_top + pad_y_bottom

    if angle == 90:
        temp_img = np.full((box_h, box_w, 3), bg_color, dtype=np.uint8)
        cv2.putText(temp_img, text, (pad_x, th + pad_y_top), font, font_scale, color, thickness)
        rotated = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_h, rot_w = rotated.shape[:2]
        
        px, py = position
        if center:
            start_x = int(px - rot_w // 2)
            start_y = int(py - rot_h // 2)
        else:
            start_x = int(px)
            start_y = int(py - rot_h)
            
        end_x = start_x + rot_w
        end_y = start_y + rot_h
        
        img_h, img_w = img.shape[:2]
        if start_y >= 0 and start_x >= 0 and end_y <= img_h and end_x <= img_w:
            img[start_y:end_y, start_x:end_x] = rotated
    else:
        x, y = position
        if center:
            x = x - tw // 2
            y = y + th // 2

        # Retângulo de fundo
        cv2.rectangle(
            img,
            (int(x - pad_x), int(y - th - pad_y_top)),
            (int(x + tw + pad_x), int(y + pad_y_bottom)),
            bg_color, -1
        )
        # Texto
        cv2.putText(img, text, (int(x), int(y)), font, font_scale, color, thickness)

def _draw_dashed_line(img, p1, p2, color, thickness, dash_length=10, gap_length=6):
    """Desenha uma linha tracejada entre p1 e p2."""
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return
    u = d / length

    dist = 0.0
    drawing = True
    while dist < length:
        seg_len = dash_length if drawing else gap_length
        end_dist = min(dist + seg_len, length)
        if drawing:
            pt_start = p1 + u * dist
            pt_end = p1 + u * end_dist
            cv2.line(
                img,
                (int(round(pt_start[0])), int(round(pt_start[1]))),
                (int(round(pt_end[0])), int(round(pt_end[1]))),
                color, thickness, cv2.LINE_AA
            )
        dist = end_dist
        drawing = not drawing


def _draw_segment_cota(
    img, p1, p2, center_pt, len_mm, font_scale, color_cota, color_text, color_bg, thickness, custom_texts=None, offset_multiplier=1.0
):
    """
    Desenha uma cota de engenharia paralela a um segmento (aresta do min_rect).
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    center = np.array(center_pt, dtype=np.float64)
    
    # Vetor direção do segmento
    v = p2 - p1
    dist_px = np.linalg.norm(v)
    if dist_px < 1e-3:
        return
        
    u = v / dist_px
    # Vetor normal
    n = np.array([-u[1], u[0]])
    
    # Direção de afastamento do centro
    mid = (p1 + p2) / 2.0
    v_c = mid - center
    if np.dot(v_c, n) < 0:
        n = -n
        
    # Deslocamento da cota em relação ao segmento original
    offset_px = int(30 * font_scale * offset_multiplier)
    tick_size = int(6 * font_scale)
    
    p1_cota = p1 + n * offset_px
    p2_cota = p2 + n * offset_px
    
    p1_cota_i = (int(round(p1_cota[0])), int(round(p1_cota[1])))
    p2_cota_i = (int(round(p2_cota[0])), int(round(p2_cota[1])))
    p1_i = (int(round(p1[0])), int(round(p1[1])))
    p2_i = (int(round(p2[0])), int(round(p2[1])))
    
    # Linhas de extensão
    ext_p1 = p1 + n * (offset_px + int(5 * font_scale))
    ext_p2 = p2 + n * (offset_px + int(5 * font_scale))
    
    ext_p1_i = (int(round(ext_p1[0])), int(round(ext_p1[1])))
    ext_p2_i = (int(round(ext_p2[0])), int(round(ext_p2[1])))
    
    # Desenhar linhas de extensão
    cv2.line(img, p1_i, ext_p1_i, color_cota, 1, cv2.LINE_AA)
    cv2.line(img, p2_i, ext_p2_i, color_cota, 1, cv2.LINE_AA)
    
    # Desenhar linha de cota principal
    cv2.line(img, p1_cota_i, p2_cota_i, color_cota, 1, cv2.LINE_AA)
    
    # Ticks arquitetônicos (traço a 45 graus)
    t1 = u + n
    t1_norm = np.linalg.norm(t1)
    if t1_norm > 1e-3:
        t1 = t1 / t1_norm
    
    t1_p1 = p1_cota + t1 * tick_size
    t1_p2 = p1_cota - t1 * tick_size
    cv2.line(img, (int(round(t1_p1[0])), int(round(t1_p1[1]))), (int(round(t1_p2[0])), int(round(t1_p2[1]))), color_cota, 2, cv2.LINE_AA)
    
    t2_p1 = p2_cota + t1 * tick_size
    t2_p2 = p2_cota - t1 * tick_size
    cv2.line(img, (int(round(t2_p1[0])), int(round(t2_p1[1]))), (int(round(t2_p2[0])), int(round(t2_p2[1]))), color_cota, 2, cv2.LINE_AA)
    
    text_angle = 90 if abs(u[0]) > abs(u[1]) else 0
    
    if custom_texts and isinstance(custom_texts, list) and len(custom_texts) == 3:
        fractions = [0.10, 0.50, 0.90]
        for i, (frac, text_val) in enumerate(zip(fractions, custom_texts)):
            # Posicionamento das três medidas (10%, 50%, 90%)
            pt_along_edge = p1 + frac * (p2 - p1)
            pt_on_cota = p1_cota + frac * (p2_cota - p1_cota)
            
            # Tick mark intermediário
            tick_p1 = pt_on_cota + t1 * tick_size
            tick_p2 = pt_on_cota - t1 * tick_size
            cv2.line(img, (int(round(tick_p1[0])), int(round(tick_p1[1]))), (int(round(tick_p2[0])), int(round(tick_p2[1]))), color_cota, 2, cv2.LINE_AA)
            
            # Linha de extensão intermediária muito suave (opcional)
            cv2.line(img, (int(round(pt_along_edge[0])), int(round(pt_along_edge[1]))), (int(round(pt_on_cota[0])), int(round(pt_on_cota[1]))), color_cota, 1, cv2.LINE_AA)
            
            # Posicionamento do texto
            text_base_pos = pt_on_cota + n * (int(20 * font_scale) if text_angle == 90 else int(12 * font_scale))
            text_pos_i = (int(round(text_base_pos[0])), int(round(text_base_pos[1])))
            
            _draw_text_with_bg(
                img, text_val, text_pos_i,
                font_scale * 0.7, color_text, color_bg, max(1, thickness - 1),
                center=True, angle=text_angle
            )
    else:
        text_val = custom_texts if isinstance(custom_texts, str) else f"{len_mm:.2f} mm"
        text_pos = mid + n * (int(20 * font_scale) if text_angle == 90 else int(12 * font_scale))
        text_pos_i = (int(round(text_pos[0])), int(round(text_pos[1])))
        
        _draw_text_with_bg(
            img, text_val, text_pos_i,
            font_scale * 0.7, color_text, color_bg, max(1, thickness - 1),
            center=True, angle=text_angle
        )


def annotate_batch(
    image_paths: list,
    metrics_list: list,
    scale: dict,
    output_dir: str = None
):
    """
    Anota um lote de imagens com seus respectivos contornos e dimensões.

    Args:
        image_paths: Lista de caminhos das imagens.
        metrics_list: Lista de dicts com métricas em mm.
        scale: Dict com fatores de escala.
        output_dir: Diretório de saída. Default: config.ANNOTATED_DIR.
    """
    output_dir = Path(output_dir or config.ANNOTATED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, metrics in zip(image_paths, metrics_list):
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"  Não foi possível carregar: {img_path}")
            continue

        output_path = output_dir / f"{img_path.stem}_annotated.png"

        # Determinar se deve desenhar elipse (peça circular)
        draw_ellipse = metrics.get("circularity", 0) > 0.80

        annotate_image(
            img, metrics, str(output_path),
            scale=scale,
            draw_ellipse=draw_ellipse,
            draw_dimensions=True
        )

    logger.info(f"  ✓ {len(image_paths)} imagens anotadas em {output_dir}")
