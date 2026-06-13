# -*- coding: utf-8 -*-
"""
Módulo de pré-processamento de imagens para o pipeline de análise cerâmica.

Responsável por:
1. Correção de distorção da lente (undistort) usando parâmetros de calibração
2. Verificação de centralização do objeto na imagem
3. Normalização de histograma (equalização no canal Y / YCrCb)
4. Filtragem gaussiana para redução de ruído

Uso:
    from preprocessing import preprocess, load_and_prepare

    gray, color = preprocess("image.tiff", "calibration.yaml")
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np

import config
from calibrate import load_calibration

logger = logging.getLogger(__name__)


def normalize_16bit_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem 16-bit para 8-bit via normalização linear.

    Mapeia o intervalo [min, max] da imagem para [0, 255], preservando
    o máximo de contraste possível da dinâmica original de 16-bit.

    Args:
        img: Imagem numpy array (uint16).

    Returns:
        Imagem normalizada (uint8).
    """
    if img.dtype == np.uint16:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img


def undistort_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coefs: np.ndarray
) -> np.ndarray:
    """
    Corrige a distorção da lente na imagem.

    Utiliza getOptimalNewCameraMatrix com alpha=1 para manter todos os pixels
    da imagem original (pode gerar bordas pretas), seguido de crop pelo ROI.

    Args:
        img: Imagem BGR (uint8 ou uint16).
        camera_matrix: Matriz intrínseca 3x3 da câmera.
        dist_coefs: Coeficientes de distorção.

    Returns:
        Imagem corrigida e recortada.
    """
    h, w = img.shape[:2]

    # Calcular nova matriz da câmera otimizada
    # alpha=1: retém todos os pixels (bordas pretas visíveis)
    # alpha=0: corta para remover bordas pretas (perde pixels das bordas)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), alpha=1, newImgSize=(w, h)
    )

    # Aplicar correção de distorção
    undistorted = cv2.undistort(img, camera_matrix, dist_coefs, None, new_camera_mtx)

    # Crop pelo ROI válido (remover bordas pretas)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        logger.debug(f"Undistort: {w}×{h} → crop {rw}×{rh} (ROI: {roi})")
    else:
        logger.warning("ROI inválido após undistort — usando imagem sem crop")

    return undistorted


def check_centering(
    img: np.ndarray,
    image_name: str = "",
    tolerance: float = None
) -> dict:
    """
    Verifica se o objeto principal está centralizado na imagem.

    A distorção radial é menor no centro da lente (especialmente na Nikon 55mm).
    Se o objeto estiver nas bordas, as medições podem ser menos precisas.

    Args:
        img: Imagem em escala de cinza (uint8).
        image_name: Nome da imagem para logging.
        tolerance: Fração da imagem considerada "zona central" (0.0 a 1.0).
                   Default: config.CENTER_TOLERANCE (0.30)

    Returns:
        Dict com:
            - center_x, center_y: Centro de massa do objeto (pixels).
            - is_centered: True se o objeto está na zona central.
            - offset_x, offset_y: Deslocamento relativo do centro (-1.0 a 1.0).
    """
    tolerance = tolerance or config.CENTER_TOLERANCE

    h, w = img.shape[:2]

    # Binarizar com Otsu para encontrar o objeto principal
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calcular centro de massa (moments)
    moments = cv2.moments(binary)
    if moments["m00"] == 0:
        logger.warning(f"  ⚠ Nenhum objeto detectado para verificação de centralização")
        return {"center_x": w // 2, "center_y": h // 2,
                "is_centered": True, "offset_x": 0.0, "offset_y": 0.0}

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Deslocamento relativo do centro da imagem (-1.0 a 1.0)
    offset_x = (cx - w / 2) / (w / 2)
    offset_y = (cy - h / 2) / (h / 2)

    # Verificar se está dentro da zona central
    is_centered = (abs(offset_x) <= tolerance) and (abs(offset_y) <= tolerance)

    if not is_centered:
        logger.warning(
            f"  ⚠ Objeto descentralizado em '{image_name}': "
            f"centro em ({cx}, {cy}), offset=({offset_x:.2f}, {offset_y:.2f}). "
            f"A distorção radial é maior nas bordas da lente 55mm. "
            f"Considere reposicionar o objeto no centro da imagem."
        )
    else:
        logger.debug(
            f"  ✓ Objeto centralizado em '{image_name}': "
            f"centro ({cx}, {cy}), offset=({offset_x:.2f}, {offset_y:.2f})"
        )

    return {
        "center_x": cx, "center_y": cy,
        "is_centered": is_centered,
        "offset_x": offset_x, "offset_y": offset_y
    }


def equalize_histogram(img_color: np.ndarray) -> np.ndarray:
    """
    Equaliza o histograma no canal de luminância para compensar variações
    de iluminação entre sessões de captura (úmida vs. seca).

    Converte BGR → YCrCb, equaliza apenas o canal Y (luminância),
    preservando a informação cromática (Cr, Cb) intacta.

    Args:
        img_color: Imagem BGR (uint8).

    Returns:
        Imagem BGR com luminância equalizada.
    """
    ycrcb = cv2.cvtColor(img_color, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def preprocess(
    image_path: str,
    calibration_yaml: str = None,
    save_undistorted: bool = True,
    output_dir: str = None
) -> tuple:
    """
    Pipeline completo de pré-processamento de uma imagem.

    Etapas:
    1. Carregar imagem (TIFF 16-bit ou 8-bit)
    2. Normalizar para 8-bit se necessário
    3. Corrigir distorção da lente (undistort)
    4. Verificar centralização do objeto
    5. Equalizar histograma (canal Y)
    6. Aplicar filtro gaussiano

    Args:
        image_path: Caminho da imagem (TIFF, PNG, JPG).
        calibration_yaml: Caminho do YAML de calibração.
                          Default: config.CALIBRATION_FILE
        save_undistorted: Se True, salva a imagem corrigida.
        output_dir: Diretório para salvar imagem corrigida.
                    Default: config.UNDISTORTED_DIR

    Returns:
        Tupla (gray_blurred, img_color):
            - gray_blurred: Imagem em escala de cinza após equalização e filtro (uint8).
            - img_color: Imagem colorida corrigida e equalizada (uint8, BGR).
              Usada para anotações visuais (desenhar contornos, dimensões).
    """
    image_path = Path(image_path)
    output_dir = output_dir or config.UNDISTORTED_DIR

    logger.info(f"Pré-processando: {image_path.name}")

    # 1. Carregar imagem
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Falha ao carregar imagem: {image_path}")

    logger.debug(f"  Carregada: {img.shape[1]}×{img.shape[0]}, dtype={img.dtype}")

    # 2. Normalizar para 8-bit se necessário
    img = normalize_16bit_to_8bit(img)

    # 3. Corrigir distorção da lente
    try:
        camera_matrix, dist_coefs = load_calibration(calibration_yaml)
        img = undistort_image(img, camera_matrix, dist_coefs)
        logger.debug("  ✓ Distorção corrigida")
    except FileNotFoundError:
        logger.warning(
            "  ⚠ Calibração não encontrada — processando sem correção de distorção. "
            "Execute 'python pipeline.py calibrate' para calibrar."
        )

    # 4. Verificar centralização
    gray_for_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check_centering(gray_for_check, image_path.name)

    # 5. Equalizar histograma no canal Y (luminância)
    img_equalized = equalize_histogram(img)
    logger.debug("  ✓ Histograma equalizado (canal Y)")

    # 6. Converter para escala de cinza e aplicar filtro gaussiano
    gray = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(
        gray, config.GAUSSIAN_KERNEL, config.GAUSSIAN_SIGMA
    )
    logger.debug(
        f"  ✓ Filtro gaussiano aplicado "
        f"(kernel={config.GAUSSIAN_KERNEL}, sigma={config.GAUSSIAN_SIGMA})"
    )

    # Salvar imagem corrigida para referência
    if save_undistorted:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_undistorted.png"
        cv2.imwrite(str(out_path), img)
        logger.debug(f"  Salva em: {out_path}")

    return gray_blurred, img_equalized
