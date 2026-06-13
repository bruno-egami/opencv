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
        # Verificar se são dados de retração ou medição individual
        if "shrinkage_width_pct" in data[0]:
            columns = CSV_SHRINKAGE_COLUMNS
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

    Returns:
        Imagem anotada.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

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

    # Bounding box
    if draw_bbox and "bbox_x" in metrics:
        x, y = metrics["bbox_x"], metrics["bbox_y"]
        bw, bh = metrics["bbox_w"], metrics["bbox_h"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), COLOR_BBOX, 2)

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

        if "bbox_w_mm" in metrics:
            texts.append(f"L: {metrics['bbox_w_mm']:.2f} mm")
        if "bbox_h_mm" in metrics:
            texts.append(f"A: {metrics['bbox_h_mm']:.2f} mm")
        if "area_mm2" in metrics:
            texts.append(f"Area: {metrics['area_mm2']:.2f} mm²")
        if "perimeter_mm" in metrics:
            texts.append(f"Perim: {metrics['perimeter_mm']:.2f} mm")
        if "circularity" in metrics:
            texts.append(f"Circ: {metrics['circularity']:.3f}")

        # Posicionar texto no canto superior esquerdo
        y_offset = 30
        for text in texts:
            _draw_text_with_bg(
                annotated, text, (10, y_offset),
                font_scale, COLOR_TEXT, COLOR_TEXT_BG, thickness
            )
            y_offset += int(35 * font_scale) + 10

    # Metadados de escala no canto inferior esquerdo
    if scale:
        meta_texts = [
            f"Vista: {scale.get('view_mode', 'N/A')}",
            f"px/mm H: {scale.get('px_per_mm_h', 0):.2f}",
            f"px/mm V: {scale.get('px_per_mm_v', 0):.2f}",
        ]
        y_offset = h - 20
        for text in reversed(meta_texts):
            _draw_text_with_bg(
                annotated, text, (10, y_offset),
                font_scale * 0.7, (200, 200, 200), COLOR_TEXT_BG, max(1, thickness - 1)
            )
            y_offset -= int(25 * font_scale) + 5

    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.debug(f"  Imagem anotada salva: {output_path}")

    return annotated


def _draw_text_with_bg(
    img, text, position, font_scale, color, bg_color, thickness
):
    """Desenha texto com fundo semi-transparente para legibilidade."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Retângulo de fundo
    cv2.rectangle(
        img,
        (x - 2, y - th - 5),
        (x + tw + 5, y + baseline + 2),
        bg_color, -1
    )
    # Texto
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


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
        draw_ellipse = metrics.get("circularity", 0) > 0.7

        annotate_image(
            img, metrics, str(output_path),
            scale=scale,
            draw_ellipse=draw_ellipse,
            draw_dimensions=True
        )

    logger.info(f"  ✓ {len(image_paths)} imagens anotadas em {output_dir}")
