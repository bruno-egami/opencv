# -*- coding: utf-8 -*-
"""
Testes unitários do pipeline de análise de corpos de prova cerâmicos.

Utiliza imagens sintéticas para validar cada módulo sem depender de
imagens reais ou hardware (câmera, checkerboard, etc.).

Executar:
    cd d:\\GitHub\\OpenCV\\ceramic_analysis
    python -m pytest tests/test_pipeline.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Adicionar diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cad_compare
from segmentation import (
    segment_background_sub,
    segment_lab,
    segment_otsu,
    segment_adaptive,
    postprocess_mask,
    extract_contour_metrics,
    evaluate_mask_quality,
    segment,
    SegmentationError,
)
from metrology import (
    calibrate_scale,
    convert_measurements,
    _sort_grid_points,
    _filter_outliers,
    _line_intersection,
    _organize_into_rows,
    _organize_into_columns,
)
from analysis import (
    calculate_shrinkage,
    compare_specimens,
    combine_views,
    export_csv,
)
from preprocessing import (
    normalize_16bit_to_8bit,
    equalize_histogram,
    check_centering,
)
from calibrate import CalibrationError


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: imagens sintéticas
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def white_rect_on_black():
    """Retângulo branco (200×100 px) centralizado sobre fundo preto (500×500)."""
    img = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(img, (150, 200), (350, 300), 255, -1)
    return img


@pytest.fixture
def white_rect_color():
    """Versão colorida do retângulo branco sobre fundo preto."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (150, 200), (350, 300), (255, 255, 255), -1)
    return img


@pytest.fixture
def circle_on_black():
    """Círculo branco (raio 80 px) centralizado sobre fundo preto."""
    img = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(img, (250, 250), 80, 255, -1)
    return img


@pytest.fixture
def low_contrast_image():
    """Retângulo cinza-claro (180) sobre fundo cinza-escuro (120) — baixo contraste."""
    img = np.full((500, 500, 3), 120, dtype=np.uint8)
    # Peça ligeiramente mais clara
    cv2.rectangle(img, (150, 200), (350, 300), (180, 175, 170), -1)
    return img


@pytest.fixture
def background_image():
    """Fundo uniforme cinza (para subtração)."""
    return np.full((500, 500, 3), 120, dtype=np.uint8)


@pytest.fixture
def image_with_object(background_image):
    """Fundo + retângulo como objeto."""
    img = background_image.copy()
    cv2.rectangle(img, (150, 200), (350, 300), (60, 70, 80), -1)
    return img


@pytest.fixture
def multi_object_image():
    """Três retângulos em fundo preto."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 200), (350, 350), (255, 255, 255), -1)
    cv2.rectangle(img, (380, 50), (480, 120), (255, 255, 255), -1)
    return img


@pytest.fixture
def synthetic_grid_points():
    """Grade sintética 5×4 com espaçamento de 50px (= 20mm → px_per_mm = 2.5)."""
    points = []
    for row in range(4):
        for col in range(5):
            x = 100 + col * 50
            y = 80 + row * 50
            points.append([x, y])
    return np.array(points, dtype=np.float32)


@pytest.fixture
def tmp_dir():
    """Diretório temporário dentro do projeto para arquivos de teste."""
    test_dir = Path(config.PROJECT_ROOT) / "tests" / "_tmp"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_dir)
    # Cleanup: remover arquivos gerados
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Pré-processamento
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_normalize_16bit_to_8bit(self):
        """16-bit → 8-bit preserva range."""
        img_16 = np.zeros((100, 100), dtype=np.uint16)
        img_16[50, 50] = 65535  # Max 16-bit
        result = normalize_16bit_to_8bit(img_16)
        assert result.dtype == np.uint8
        assert result[50, 50] == 255
        assert result[0, 0] == 0

    def test_normalize_8bit_passthrough(self):
        """8-bit passa sem modificação."""
        img_8 = np.full((100, 100), 128, dtype=np.uint8)
        result = normalize_16bit_to_8bit(img_8)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img_8)

    def test_equalize_histogram_preserves_shape(self):
        """Equalização preserva dimensões e tipo."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = equalize_histogram(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_check_centering_centered(self, white_rect_on_black):
        """Objeto centralizado retorna is_centered=True."""
        result = check_centering(white_rect_on_black)
        assert result["is_centered"] is True
        assert abs(result["offset_x"]) < 0.3
        assert abs(result["offset_y"]) < 0.3

    def test_check_centering_off_center(self):
        """Objeto no canto retorna is_centered=False."""
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (50, 50), 255, -1)  # Canto superior esquerdo
        result = check_centering(img)
        assert result["is_centered"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Segmentação
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentation:

    def test_segment_otsu_dark_on_light(self):
        """Otsu segmenta objeto escuro sobre fundo claro (simula argila sobre MDF).

        segment_otsu usa THRESH_BINARY_INV: objeto escuro → branco na máscara.
        Isso corresponde ao cenário real (argila escura sobre MDF claro).
        """
        # Fundo claro (MDF) com retângulo escuro (argila)
        img = np.full((500, 500), 200, dtype=np.uint8)
        cv2.rectangle(img, (150, 200), (350, 300), 60, -1)

        mask = segment_otsu(img)
        assert mask.dtype == np.uint8
        assert cv2.countNonZero(mask) > 0

        # O retângulo escuro deve ser branco na máscara (INV)
        assert mask[250, 250] == 255  # Centro do retângulo (objeto)
        assert mask[10, 10] == 0  # Canto (fundo claro)

    def test_segment_background_sub(self, background_image, image_with_object):
        """Subtração de fundo isola o objeto."""
        mask = segment_background_sub(image_with_object, background_image, threshold=20)
        assert cv2.countNonZero(mask) > 0

        # O objeto deve estar na máscara
        assert mask[250, 250] == 255
        # Fundo fora do objeto deve ser preto
        assert mask[10, 10] == 0

    def test_segment_lab(self, low_contrast_image):
        """LAB segmenta objeto de baixo contraste."""
        mask = segment_lab(low_contrast_image)
        assert mask.dtype == np.uint8
        # Deve haver alguma segmentação (pode não ser perfeita com contraste tão baixo)
        assert mask.shape == low_contrast_image.shape[:2]

    def test_segment_adaptive(self, white_rect_on_black):
        """Threshold adaptativo funciona."""
        mask = segment_adaptive(white_rect_on_black)
        assert mask.dtype == np.uint8
        assert cv2.countNonZero(mask) > 0

    def test_postprocess_mask_closes_gaps(self):
        """MORPH_CLOSE fecha lacunas em bordas."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        # Criar uma lacuna
        cv2.rectangle(mask, (95, 95), (105, 105), 0, -1)

        closed = postprocess_mask(mask)

        # A lacuna deve estar fechada
        assert closed[100, 100] == 255

    def test_evaluate_mask_quality_good(self, white_rect_on_black):
        """Máscara com retângulo claro tem boa qualidade."""
        quality = evaluate_mask_quality(white_rect_on_black)
        assert quality["is_good"] is True
        assert quality["n_contours"] >= 1
        assert quality["proportion"] > 0.01

    def test_evaluate_mask_quality_empty(self):
        """Máscara vazia tem qualidade ruim."""
        empty = np.zeros((500, 500), dtype=np.uint8)
        quality = evaluate_mask_quality(empty)
        assert quality["is_good"] is False
        assert quality["n_contours"] == 0

    def test_extract_contour_metrics_rect(self, white_rect_on_black):
        """Métricas de retângulo têm valores esperados."""
        contours, _ = cv2.findContours(
            white_rect_on_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(contours) >= 1

        metrics = extract_contour_metrics(contours[0])

        # Retângulo 200×100 px
        assert abs(metrics["bbox_w"] - 200) <= 2
        assert abs(metrics["bbox_h"] - 100) <= 2
        assert metrics["area_px"] > 19000  # ~200*100 = 20000
        assert metrics["circularity"] < 0.9  # Retângulo, não círculo

    def test_extract_contour_metrics_circle(self, circle_on_black):
        """Círculo tem circularidade alta."""
        contours, _ = cv2.findContours(
            circle_on_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        metrics = extract_contour_metrics(contours[0])
        assert metrics["circularity"] > 0.85  # Quase circular

    def test_segment_multi_contour(self, multi_object_image):
        """Múltiplos objetos geram múltiplos contornos."""
        gray = cv2.cvtColor(multi_object_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        results = segment(
            blurred, multi_object_image, "multi_test",
            save_mask=False
        )

        assert len(results) == 3  # 3 retângulos

    def test_segment_empty_raises(self):
        """Imagem sem objeto lança SegmentationError."""
        empty_gray = np.zeros((500, 500), dtype=np.uint8)
        empty_color = np.zeros((500, 500, 3), dtype=np.uint8)

        with pytest.raises(SegmentationError):
            segment(empty_gray, empty_color, "empty_test", save_mask=False)


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Metrologia
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrology:

    def test_calibrate_scale_known_spacing(self, synthetic_grid_points):
        """Grade com espaçamento conhecido produz px_per_mm correto."""
        # Grade 5×4, espaçamento 50px, grid_spacing_mm = 20mm
        # → px_per_mm = 50 / 20 = 2.5
        scale = calibrate_scale(
            synthetic_grid_points,
            view_mode="top",
            grid_spacing_mm=20.0
        )

        assert abs(scale["px_per_mm_h"] - 2.5) < 0.1
        assert abs(scale["px_per_mm_v"] - 2.5) < 0.1
        assert scale["anisotropy"] < 0.02  # Isotrópico
        assert scale["view_mode"] == "top"

    def test_calibrate_scale_anisotropic(self):
        """Detecta anisotropia quando espaçamentos H e V diferem."""
        # Grade com espaçamento diferente em H (50px) e V (60px)
        points = []
        for row in range(4):
            for col in range(5):
                x = 100 + col * 50
                y = 80 + row * 60  # V diferente
                points.append([x, y])
        points = np.array(points, dtype=np.float32)

        scale = calibrate_scale(points, view_mode="top", grid_spacing_mm=20.0)

        assert abs(scale["px_per_mm_h"] - 2.5) < 0.1   # 50/20
        assert abs(scale["px_per_mm_v"] - 3.0) < 0.1    # 60/20
        assert scale["anisotropy"] > 0.1  # Detectada

    def test_convert_measurements(self, synthetic_grid_points):
        """Conversão px → mm com fator conhecido."""
        scale = {
            "px_per_mm_h": 2.5,
            "px_per_mm_v": 2.5,
            "anisotropy": 0.0,
            "view_mode": "top",
        }

        metrics_px = {
            "bbox_w": 250,   # 250 / 2.5 = 100 mm
            "bbox_h": 125,   # 125 / 2.5 = 50 mm
            "area_px": 31250,  # 31250 / (2.5*2.5) = 5000 mm²
            "perimeter_px": 750,  # 750 / 2.5 = 300 mm
            "min_rect_w": 250,
            "min_rect_h": 125,
            "ellipse_major_px": 250,
            "ellipse_minor_px": 125,
        }

        metrics_mm = convert_measurements(metrics_px, scale)

        assert abs(metrics_mm["bbox_w_mm"] - 100.0) < 0.1
        assert abs(metrics_mm["bbox_h_mm"] - 50.0) < 0.1
        assert abs(metrics_mm["area_mm2"] - 5000.0) < 1.0
        assert abs(metrics_mm["perimeter_mm"] - 300.0) < 0.1
        assert abs(metrics_mm["min_rect_w_mm"] - 100.0) < 0.1
        assert abs(metrics_mm["min_rect_h_mm"] - 50.0) < 0.1
        assert abs(metrics_mm["ellipse_major_mm"] - 100.0) < 0.1
        assert abs(metrics_mm["ellipse_minor_mm"] - 50.0) < 0.1

        # Caso anisotrópico e rotacionado
        scale_aniso = {
            "px_per_mm_h": 2.0,
            "px_per_mm_v": 4.0,
            "anisotropy": 0.5,
            "view_mode": "top",
        }

        # Orientação a -90 graus (como o espécime do usuário)
        metrics_px_rotated = {
            "bbox_w": 200,
            "bbox_h": 400,
            "area_px": 80000,
            "perimeter_px": 1200,
            "min_rect_w": 400,   # orientação vertical -> deve usar px_per_mm_v = 4.0
            "min_rect_h": 200,   # orientação horizontal -> deve usar px_per_mm_h = 2.0
            "min_rect_angle": -90.0,
            "ellipse_major_px": 400,
            "ellipse_minor_px": 200,
            "ellipse_angle": -90.0,
        }

        metrics_mm_aniso = convert_measurements(metrics_px_rotated, scale_aniso)
        # min_rect_w_mm = 400 / 4.0 = 100.0
        assert abs(metrics_mm_aniso["min_rect_w_mm"] - 100.0) < 0.1
        # min_rect_h_mm = 200 / 2.0 = 100.0
        assert abs(metrics_mm_aniso["min_rect_h_mm"] - 100.0) < 0.1
        # ellipse_major_mm = 400 / 4.0 = 100.0
        assert abs(metrics_mm_aniso["ellipse_major_mm"] - 100.0) < 0.1
        # ellipse_minor_mm = 200 / 2.0 = 100.0
        assert abs(metrics_mm_aniso["ellipse_minor_mm"] - 100.0) < 0.1

    def test_sort_grid_points(self):
        """Pontos desordenados são organizados em ordem de leitura."""
        # Pontos em ordem aleatória
        points = np.array([
            [300, 100], [100, 100], [200, 100],
            [300, 200], [100, 200], [200, 200],
        ], dtype=np.float32)

        sorted_pts = _sort_grid_points(points)

        # Primeira linha (Y=100), ordenada por X
        assert sorted_pts[0, 0] < sorted_pts[1, 0] < sorted_pts[2, 0]
        # Segunda linha (Y=200)
        assert sorted_pts[3, 0] < sorted_pts[4, 0] < sorted_pts[5, 0]

    def test_filter_outliers(self):
        """Outliers são removidos."""
        values = np.array([50, 50, 51, 49, 50, 200, 50])  # 200 é outlier
        filtered = _filter_outliers(values)
        assert 200 not in filtered
        assert len(filtered) < len(values)

    def test_line_intersection(self):
        """Calcula interseção de duas linhas."""
        line1 = ((0, 50), (100, 50))   # Horizontal em Y=50
        line2 = ((50, 0), (50, 100))   # Vertical em X=50

        point = _line_intersection(line1, line2)
        assert point is not None
        assert abs(point[0] - 50) < 0.01
        assert abs(point[1] - 50) < 0.01

    def test_line_intersection_parallel(self):
        """Linhas paralelas retornam None."""
        line1 = ((0, 50), (100, 50))
        line2 = ((0, 100), (100, 100))
        assert _line_intersection(line1, line2) is None

    def test_organize_into_rows(self, synthetic_grid_points):
        """Pontos são organizados em linhas corretas."""
        rows = _organize_into_rows(synthetic_grid_points)
        assert len(rows) == 4  # 4 linhas na grade sintética

    def test_organize_into_columns(self, synthetic_grid_points):
        """Pontos são organizados em colunas corretas."""
        cols = _organize_into_columns(synthetic_grid_points)
        assert len(cols) == 5  # 5 colunas na grade sintética


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Análise
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalysis:

    def test_shrinkage_calculation_basic(self):
        """ΔL% = (100 - 90) / 100 × 100 = 10%."""
        result = calculate_shrinkage(100.0, 90.0)
        assert abs(result - 10.0) < 0.001

    def test_shrinkage_calculation_zero(self):
        """Sem variação → retração 0%."""
        result = calculate_shrinkage(100.0, 100.0)
        assert abs(result) < 0.001

    def test_shrinkage_calculation_expansion(self):
        """Expansão (raro) → retração negativa."""
        result = calculate_shrinkage(100.0, 110.0)
        assert result < 0  # -10%

    def test_shrinkage_zero_wet(self):
        """Valor úmido zero → retorna 0 (sem divisão por zero)."""
        result = calculate_shrinkage(0.0, 50.0)
        assert result == 0.0

    def test_compare_specimens_basic(self):
        """Comparação de pares úmido-seco funciona."""
        wet = [
            {"sample_id": "A", "bbox_w_mm": 100.0, "bbox_h_mm": 50.0,
             "area_mm2": 5000.0, "perimeter_mm": 300.0,
             "ellipse_major_mm": 0, "ellipse_minor_mm": 0,
             "min_rect_w_mm": 100, "min_rect_h_mm": 50,
             "circularity": 0.5, "px_per_mm_h": 2.5, "px_per_mm_v": 2.5,
             "anisotropy": 0.0}
        ]
        dry = [
            {"sample_id": "A", "bbox_w_mm": 90.0, "bbox_h_mm": 45.0,
             "area_mm2": 4050.0, "perimeter_mm": 270.0,
             "ellipse_major_mm": 0, "ellipse_minor_mm": 0,
             "min_rect_w_mm": 90, "min_rect_h_mm": 45,
             "circularity": 0.5, "px_per_mm_h": 2.5, "px_per_mm_v": 2.5,
             "anisotropy": 0.0}
        ]

        results = compare_specimens(wet, dry)
        assert len(results) == 1
        assert abs(results[0]["shrinkage_width_pct"] - 10.0) < 0.1
        assert abs(results[0]["shrinkage_height_pct"] - 10.0) < 0.1

    def test_compare_specimens_no_match(self):
        """Sem pares correspondentes, retorna dados individuais."""
        wet = [{"sample_id": "A", "image_name": "A"}]
        dry = [{"sample_id": "B", "image_name": "B"}]

        results = compare_specimens(wet, dry)
        assert len(results) == 2  # Dados individuais, sem retração

    def test_combine_views(self):
        """Combinação top + side produz dimensões 3D."""
        top = [{"sample_id": "A", "bbox_w_mm": 100, "bbox_h_mm": 60}]
        side = [{"sample_id": "A", "bbox_w_mm": 100, "bbox_h_mm": 40}]

        combined = combine_views(top, side)
        assert len(combined) == 1
        assert combined[0]["width_mm"] == 100   # X (do top)
        assert combined[0]["depth_mm"] == 60    # Y (do top)
        assert combined[0]["height_mm"] == 40   # Z (do side)

    def test_combine_views_cross_validation(self):
        """Largura deve ser consistente entre vistas."""
        top = [{"sample_id": "A", "bbox_w_mm": 100}]
        side = [{"sample_id": "A", "bbox_w_mm": 95, "bbox_h_mm": 40}]

        combined = combine_views(top, side)
        # Diferença de 5% deve ser reportada
        assert combined[0].get("width_cross_validation_pct", 0) > 0

    def test_export_csv(self, tmp_dir):
        """CSV é exportado com todas as colunas."""
        data = [
            {
                "sample_id": "A",
                "session": "test",
                "view_mode": "top",
                "state": "wet",
                "bbox_w_mm": 100.0,
                "bbox_h_mm": 50.0,
                "area_mm2": 5000.0,
            }
        ]

        csv_path = os.path.join(tmp_dir, "test_results.csv")
        export_csv(data, csv_path)

        assert os.path.exists(csv_path)

        # Verificar conteúdo
        with open(csv_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 row
            header = lines[0].strip()
            assert "sample_id" in header
            assert "bbox_w_mm" in header

    def test_export_csv_empty(self, tmp_dir):
        """CSV vazio não gera erro."""
        csv_path = os.path.join(tmp_dir, "empty.csv")
        export_csv([], csv_path)
        # Não deve ter criado arquivo
        assert not os.path.exists(csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Calibração
# ──────────────────────────────────────────────────────────────────────────────

class TestCalibration:

    def test_calibration_error_raised(self):
        """CalibrationError pode ser criada e capturada."""
        with pytest.raises(CalibrationError):
            raise CalibrationError("Teste")

    def test_load_calibration_missing_file(self):
        """Arquivo de calibração inexistente lança FileNotFoundError."""
        from calibrate import load_calibration
        with pytest.raises(FileNotFoundError):
            load_calibration("/path/that/does/not/exist.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Testes: Integração parcial
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_segmentation_pipeline(self, white_rect_color, tmp_dir):
        """Pipeline completo: pré-proc → segmentação → métricas."""
        gray = cv2.cvtColor(white_rect_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        results = segment(
            blurred, white_rect_color, "integration_test",
            save_mask=True,
            masks_dir=tmp_dir
        )

        assert len(results) >= 1
        metrics = results[0]

        # Verificar métricas do retângulo 200×100
        assert abs(metrics["bbox_w"] - 200) <= 5
        assert abs(metrics["bbox_h"] - 100) <= 5
        assert metrics["area_px"] > 18000

        # Verificar máscara salva
        mask_path = os.path.join(tmp_dir, "integration_test_mask.png")
        assert os.path.exists(mask_path)

    def test_full_measurement_pipeline(self, white_rect_color, synthetic_grid_points):
        """Pipeline: segmentação → escala → conversão mm."""
        gray = cv2.cvtColor(white_rect_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        results = segment(blurred, white_rect_color, "measure_test", save_mask=False)
        metrics_px = results[0]

        scale = calibrate_scale(
            synthetic_grid_points, view_mode="top", grid_spacing_mm=20.0
        )

        metrics_mm = convert_measurements(metrics_px, scale)

        # Com px_per_mm ≈ 2.5:
        # bbox_w ≈ 200 px → 200/2.5 = 80 mm
        # bbox_h ≈ 100 px → 100/2.5 = 40 mm
        assert abs(metrics_mm["bbox_w_mm"] - 80.0) < 5.0
        assert abs(metrics_mm["bbox_h_mm"] - 40.0) < 5.0

    def test_background_sub_workflow(
        self, background_image, image_with_object, tmp_dir
    ):
        """Workflow completo com subtração de fundo."""
        gray = cv2.cvtColor(image_with_object, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        results = segment(
            blurred, image_with_object, "bg_sub_test",
            background=background_image,
            strategy="background_sub",
            save_mask=True,
            masks_dir=tmp_dir
        )

        assert len(results) >= 1
        assert results[0]["area_px"] > 0

    def test_shrinkage_end_to_end(self):
        """Teste end-to-end: criar peças, medir, calcular retração."""
        # Simular peça úmida (maior) e seca (menor)
        wet_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(wet_img, (100, 150), (400, 350), (255, 255, 255), -1)
        # 300 × 200 px

        dry_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(dry_img, (120, 165), (380, 335), (255, 255, 255), -1)
        # 260 × 170 px

        # Escala fixa: 2.5 px/mm
        scale = {
            "px_per_mm_h": 2.5, "px_per_mm_v": 2.5,
            "anisotropy": 0.0, "view_mode": "top",
        }

        # Segmentar ambas
        wet_gray = cv2.cvtColor(wet_img, cv2.COLOR_BGR2GRAY)
        dry_gray = cv2.cvtColor(dry_img, cv2.COLOR_BGR2GRAY)

        wet_results = segment(
            cv2.GaussianBlur(wet_gray, (5, 5), 0),
            wet_img, "sample_wet", save_mask=False
        )
        dry_results = segment(
            cv2.GaussianBlur(dry_gray, (5, 5), 0),
            dry_img, "sample_dry", save_mask=False
        )

        # Converter para mm
        wet_mm = convert_measurements(wet_results[0], scale)
        wet_mm["sample_id"] = "sample"
        wet_mm["state"] = "wet"

        dry_mm = convert_measurements(dry_results[0], scale)
        dry_mm["sample_id"] = "sample"
        dry_mm["state"] = "dry"

        # Comparar
        comparisons = compare_specimens([wet_mm], [dry_mm])
        assert len(comparisons) == 1

        # Retração esperada:
        # Largura: (120 - 104) / 120 × 100 ≈ 13.3%
        # (300px→260px → 120mm→104mm)
        shrinkage_w = comparisons[0]["shrinkage_width_pct"]
        assert 10 < shrinkage_w < 20  # Faixa esperada

    def test_auto_inversion_detection(self):
        """Testa se uma máscara invertida (objeto preto no fundo branco) é auto-corrigida."""
        from segmentation import check_and_correct_inversion
        # Criar máscara invertida (fundo branco=255, objeto preto=0 no centro)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (70, 70), 0, -1)

        corrected = check_and_correct_inversion(mask)

        # Deve ser invertida (objeto branco no fundo preto)
        assert corrected[0, 0] == 0
        assert corrected[50, 50] == 255

    def test_contour_sorting_by_score(self):
        """Testa se múltiplos contornos são ordenados pelo score (área × proximidade do centro)."""
        # Três retângulos com áreas de 10.000 px² (> 5.000 px² de limiar mínimo)
        img = np.zeros((300, 800), dtype=np.uint8)
        cv2.rectangle(img, (500, 100), (600, 200), 255, -1)  # Centroide X ≈ 550
        cv2.rectangle(img, (100, 100), (200, 200), 255, -1)  # Centroide X ≈ 150
        cv2.rectangle(img, (300, 100), (400, 200), 255, -1)  # Centroide X ≈ 350

        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = segment(img, color, "sort_test", save_mask=False)

        assert len(results) == 3
        # Devem estar ordenados pelo score: mais perto do centro (300), depois (500), depois (100)
        x_coords = [r["bbox_x"] for r in results]
        assert x_coords == [300, 500, 100]
        assert results[0]["contour_index"] == 0
        assert results[1]["contour_index"] == 1
        assert results[2]["contour_index"] == 2

    def test_parallax_correction_formula_precision(self):
        """Testa a nova fórmula exata de correção de paralaxe baseada em focal_px/px_per_mm."""
        from metrology import _apply_parallax_correction
        
        # Simular matriz da câmera com focal de 1000px
        camera_matrix = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        px_per_mm_h = 2.0
        px_per_mm_v = 2.0
        
        # dist_grade_mm = 1000 / 2.0 = 500.0 mm
        # Se gap = 5mm:
        # dist_peca_mm = 495.0 mm
        # correction = 500.0 / 495.0 = 1.010101
        
        # Precisamos temporariamente mockar o gap no config
        original_gap = config.SIDE_GRID_GAP_MM
        config.SIDE_GRID_GAP_MM = 5.0
        try:
            h_corr, v_corr = _apply_parallax_correction(
                px_per_mm_h, px_per_mm_v, camera_matrix=camera_matrix
            )
            expected_corr = 500.0 / 495.0
            assert abs(h_corr - px_per_mm_h * expected_corr) < 1e-5
            assert abs(v_corr - px_per_mm_v * expected_corr) < 1e-5
        finally:
            config.SIDE_GRID_GAP_MM = original_gap


class TestCadComparison:
    def test_load_stl(self):
        import trimesh
        # Cria um cubo 40x20x15 sintético
        box = trimesh.creation.box((40, 20, 15))
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            box.export(tmp_path)
            mesh = cad_compare.load_cad_model(tmp_path)
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_step_extension(self, monkeypatch):
        import trimesh
        # Mock de _load_step para testar o direcionamento de extensão sem precisar de arquivo STEP real
        called = False
        def mock_load_step(filepath, tolerance, angular_tolerance):
            nonlocal called
            called = True
            return trimesh.creation.box((10, 10, 10))
            
        monkeypatch.setattr(cad_compare, "_load_step", mock_load_step)
        
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(b"dummy step content")
            mesh = cad_compare.load_cad_model(tmp_path)
            assert called
            assert len(mesh.vertices) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_detect_orientation_prism(self):
        import trimesh
        box = trimesh.creation.box((40, 20, 15))
        orientation = cad_compare.detect_orientation(box)
        assert orientation["shape_class"] == "prismatic"
        assert not orientation["is_symmetric"]
        
    def test_detect_orientation_cylinder(self):
        import trimesh
        cyl = trimesh.creation.cylinder(radius=10, height=30)
        orientation = cad_compare.detect_orientation(cyl)
        assert orientation["shape_class"] == "axisymmetric"
        assert orientation["is_symmetric"]
        
    def test_detect_orientation_organic(self):
        import trimesh
        sph = trimesh.creation.icosphere(subdivisions=2, radius=10)
        orientation = cad_compare.detect_orientation(sph)
        assert orientation["shape_class"] == "organic"

    def test_align_mesh_rotated(self):
        import trimesh
        box = trimesh.creation.box((40, 20, 15))
        # Rotação de 45 graus sobre Z
        rad = np.radians(45)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.eye(4)
        rot[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        box.apply_transform(rot)
        
        orientation = cad_compare.detect_orientation(box)
        aligned = cad_compare.align_mesh(box, orientation)
        
        extents = aligned.bounds[1] - aligned.bounds[0]
        sorted_extents = sorted(extents)
        assert abs(sorted_extents[0] - 15) < 1.0
        assert abs(sorted_extents[1] - 20) < 1.0
        assert abs(sorted_extents[2] - 40) < 1.0

    def test_project_top_prism(self):
        import trimesh
        box = trimesh.creation.box((40, 20, 15))
        ext, holes, mask, bbox = cad_compare.project_to_2d(box, "top")
        assert abs(bbox["width_mm"] - 40) < 1e-3
        assert abs(bbox["height_mm"] - 20) < 1e-3
        assert len(ext) > 0
        assert len(holes) == 0
        
    def test_project_front_prism(self):
        import trimesh
        box = trimesh.creation.box((40, 20, 15))
        ext, holes, mask, bbox = cad_compare.project_to_2d(box, "front")
        assert abs(bbox["width_mm"] - 40) < 1e-3
        assert abs(bbox["height_mm"] - 15) < 1e-3
        
    def test_project_left_prism(self):
        import trimesh
        box = trimesh.creation.box((40, 20, 15))
        ext, holes, mask, bbox = cad_compare.project_to_2d(box, "left")
        assert abs(bbox["width_mm"] - 20) < 1e-3
        assert abs(bbox["height_mm"] - 15) < 1e-3
        
    def test_project_top_cylinder(self):
        import trimesh
        cyl = trimesh.creation.cylinder(radius=10, height=30)
        ext, holes, mask, bbox = cad_compare.project_to_2d(cyl, "top")
        assert abs(bbox["width_mm"] - 20) < 0.5
        assert abs(bbox["height_mm"] - 20) < 0.5
        
    def test_project_with_holes(self, monkeypatch):
        import trimesh
        from shapely.geometry import Polygon as ShapelyPolygon
        
        ext_coords = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole_coords = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly_with_hole = ShapelyPolygon(ext_coords, [hole_coords])
        
        class MockPath2D:
            def __init__(self, polygons):
                self.polygons_full = polygons
                
        mock_path = MockPath2D([poly_with_hole])
        monkeypatch.setattr(trimesh.Trimesh, "projected", lambda self, normal: mock_path)
        
        box = trimesh.creation.box((20, 20, 20))
        ext, holes, mask, bbox = cad_compare.project_to_2d(box, "top")
        
        assert len(holes) == 1
        min_h = np.min(holes[0], axis=0)
        max_h = np.max(holes[0], axis=0)
        assert np.allclose(min_h, [5, 5])
        assert np.allclose(max_h, [15, 15])

    def test_resample_contour(self):
        rect = np.array([(0, 0), (40, 0), (40, 20), (0, 20)])
        resampled = cad_compare.resample_contour(rect, target_spacing_mm=1.0)
        assert 115 <= len(resampled) <= 125
        
        pts = np.vstack([resampled, resampled[0]])
        dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        assert np.all(dists < 1.1)
        
    def test_shape_complexity(self):
        angles = np.linspace(0, 2*np.pi, 200, endpoint=False)
        circle = np.column_stack([np.cos(angles), np.sin(angles)]) * 10.0
        comp_circle = cad_compare.compute_shape_complexity(circle)
        assert abs(comp_circle - 12.566) < 0.2
        
        square = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        comp_sq = cad_compare.compute_shape_complexity(square)
        assert abs(comp_sq - 16.0) < 0.1

    def test_register_centroid(self):
        sq1 = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        sq2 = sq1 + [5, -3]
        
        aligned, transform = cad_compare.register_contours(sq1, sq2, method="centroid")
        assert np.allclose(transform["translation_mm"], [5, -3])
        assert np.allclose(aligned, sq2)
        
    def test_register_icp_rotation(self):
        # Retângulo de 20x10 para evitar simetria quadrada de 90°
        r1 = np.array([(0, 0), (20, 0), (20, 10), (0, 10)])
        # Rotacionado 90° em relação à origem: (x, y) -> (-y, x)
        r2 = np.array([(0, 0), (0, 20), (-10, 20), (-10, 0)])
        
        aligned, transform = cad_compare.register_contours(r1, r2, method="icp", shape_class="prismatic")
        # Espera-se rotação de 90° (ou -270°)
        assert abs(transform["rotation_deg"] - 90.0) < 5.0 or abs(transform["rotation_deg"] + 270.0) < 5.0
        
    def test_register_symmetric_no_rotation(self):
        angles = np.linspace(0, 2*np.pi, 50, endpoint=False)
        c1 = np.column_stack([np.cos(angles), np.sin(angles)]) * 10.0
        c2 = np.column_stack([np.cos(angles + 0.5), np.sin(angles + 0.5)]) * 10.0
        
        aligned, transform = cad_compare.register_contours(c1, c2, method="icp", shape_class="axisymmetric")
        assert transform["rotation_deg"] == 0.0
        
    def test_compare_identical(self):
        sq = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        bbox = {"width_mm": 10.0, "height_mm": 10.0, "area_mm2": 100.0}
        metrics = cad_compare.compare_contours(sq, sq, bbox, shape_class="prismatic")
        
        assert abs(metrics["hausdorff_mm"]) < 1e-5
        assert abs(metrics["mean_deviation_mm"]) < 1e-5
        assert abs(metrics["iou"] - 1.0) < 1e-5
        assert abs(metrics["bbox_w_deviation_mm"]) < 1e-5
        
    def test_compare_scaled(self):
        sq1 = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        sq2 = sq1 * 0.95
        bbox = {"width_mm": 10.0, "height_mm": 10.0, "area_mm2": 100.0}
        metrics = cad_compare.compare_contours(sq1, sq2, bbox, shape_class="prismatic")
        
        assert abs(metrics["bbox_w_deviation_mm"] + 0.5) < 0.1
        assert abs(metrics["bbox_w_deviation_pct"] + 5.0) < 1.0
        
    def test_deviation_map_output(self):
        photo = np.zeros((100, 100, 3), dtype=np.uint8)
        cad = np.array([(10, 10), (90, 10), (90, 90), (10, 90)])
        photo_c = cad + 1.0
        dists = np.ones(len(cad)) * 1.0
        
        metrics = {
            "hausdorff_mm": 1.0,
            "mean_deviation_mm": 1.0,
            "deviation_std_mm": 0.0,
            "iou": 0.95,
            "shape_complexity": 16.0,
            "bbox_w_deviation_mm": 0.0,
            "bbox_w_deviation_pct": 0.0,
            "bbox_h_deviation_mm": 0.0,
            "bbox_h_deviation_pct": 0.0,
            "area_deviation_pct": 0.0
        }
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cad_compare.generate_deviation_map(
                photo, cad, photo_c, dists, tmp_path,
                px_per_mm_h=2.0, px_per_mm_v=2.0,
                tolerance_mm=1.0, metrics=metrics
            )
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def test_find_checkerboard_mocked_interactive(monkeypatch):
    """Testa find_checkerboard com interacao manual mockada."""
    import calibrate
    import interactive
    
    img = np.zeros((100, 100), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cv2.imwrite(tmp_path, img)
        
        # 1. Mockar as funcoes interativas
        mock_corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        monkeypatch.setattr(interactive, "select_checkerboard_corners_manually", lambda *args, **kwargs: mock_corners)
        monkeypatch.setattr(interactive, "fine_tune_checkerboard_grid", lambda img_val, pts, *args, **kwargs: (pts, True))
        
        # 2. Simular que nao estamos no pytest para forcar use_interactive a ser True
        original_modules = sys.modules.copy()
        if "pytest" in sys.modules:
            del sys.modules["pytest"]
            
        try:
            monkeypatch.setattr(config, "INTERACTIVE_CALIBRATION", True)
            found, corners, gray = calibrate.find_checkerboard(tmp_path, (14, 10))
            
            assert found is True
            assert corners is not None
            assert corners.shape == (140, 1, 2)
        finally:
            sys.modules.update(original_modules)
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
