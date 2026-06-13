# -*- coding: utf-8 -*-
"""
Módulo de conversão de arquivos RAW (.NEF Nikon) para TIFF 16-bit.

Preserva a profundidade de cor total do sensor (65.536 níveis por canal),
crucial para distinguir tonalidades próximas como argila marrom sobre MDF.

Uso:
    from raw_converter import convert_raw, convert_batch

    # Converter um único arquivo
    output_path = convert_raw("foto.NEF", "output_dir/")

    # Converter todos os .NEF de um diretório
    converted = convert_batch("raw_dir/", "output_dir/")
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None

import config

logger = logging.getLogger(__name__)


def check_rawpy_available():
    """Verifica se o rawpy está instalado."""
    if rawpy is None:
        raise ImportError(
            "O módulo 'rawpy' é necessário para conversão de arquivos RAW.\n"
            "Instale com: pip install rawpy"
        )


def convert_raw(nef_path: str, output_dir: str) -> str:
    """
    Converte um arquivo RAW (.NEF) para TIFF 16-bit.

    A conversão utiliza o white balance da câmera e desabilita o brilho
    automático para preservar a linearidade radiométrica do sensor.

    Args:
        nef_path: Caminho do arquivo .NEF de entrada.
        output_dir: Diretório onde salvar o TIFF convertido.

    Returns:
        Caminho completo do TIFF gerado.

    Raises:
        ImportError: Se rawpy não estiver instalado.
        FileNotFoundError: Se o arquivo .NEF não existir.
        RuntimeError: Se a conversão falhar.
    """
    check_rawpy_available()

    nef_path = Path(nef_path)
    if not nef_path.exists():
        raise FileNotFoundError(f"Arquivo RAW não encontrado: {nef_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = nef_path.stem + ".tiff"
    output_path = output_dir / output_filename

    logger.info(f"Convertendo {nef_path.name} → {output_filename}")

    try:
        with rawpy.imread(str(nef_path)) as raw:
            # Pós-processamento do RAW com configurações otimizadas:
            rgb = raw.postprocess(
                use_camera_wb=config.USE_CAMERA_WB,  # WB registrado pela Nikon
                half_size=False,           # Resolução total do sensor (2592×3872)
                no_auto_bright=True,       # Sem brilho automático (linearidade)
                output_bps=config.OUTPUT_BPS,  # 16 bits por canal
                output_color=rawpy.ColorSpace.sRGB,  # Espaço de cor padrão
            )

        # rawpy retorna RGB; OpenCV utiliza BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Salvar como TIFF 16-bit (sem compressão, preserva dados)
        success = cv2.imwrite(str(output_path), bgr)
        if not success:
            raise RuntimeError(f"Falha ao salvar TIFF: {output_path}")

        logger.info(
            f"  ✓ {output_filename} — "
            f"{bgr.shape[1]}×{bgr.shape[0]} px, "
            f"{bgr.dtype}, "
            f"{output_path.stat().st_size / 1024 / 1024:.1f} MB"
        )
        return str(output_path)

    except Exception as e:
        logger.error(f"Erro ao converter {nef_path.name}: {e}")
        raise


def convert_batch(input_dir: str, output_dir: str) -> list:
    """
    Converte todos os arquivos RAW (.NEF) de um diretório para TIFF 16-bit.

    Args:
        input_dir: Diretório contendo arquivos .NEF.
        output_dir: Diretório de saída para os TIFFs.

    Returns:
        Lista de caminhos dos TIFFs gerados.

    Raises:
        FileNotFoundError: Se o diretório de entrada não existir.
        ValueError: Se nenhum arquivo RAW for encontrado.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")

    # Encontrar todos os arquivos RAW
    raw_files = []
    for ext in config.RAW_EXTENSIONS:
        raw_files.extend(input_dir.glob(f"*{ext}"))
    raw_files.sort()

    if not raw_files:
        raise ValueError(
            f"Nenhum arquivo RAW encontrado em {input_dir}\n"
            f"Extensões procuradas: {config.RAW_EXTENSIONS}"
        )

    logger.info(f"Encontrados {len(raw_files)} arquivos RAW em {input_dir}")

    converted = []
    errors = []
    for nef_path in raw_files:
        try:
            tiff_path = convert_raw(str(nef_path), output_dir)
            converted.append(tiff_path)
        except Exception as e:
            errors.append((str(nef_path), str(e)))
            logger.error(f"  ✗ Falha ao converter {nef_path.name}: {e}")

    logger.info(
        f"Conversão concluída: {len(converted)} sucesso, {len(errors)} falhas"
    )

    if errors:
        logger.warning("Arquivos com falha na conversão:")
        for path, error in errors:
            logger.warning(f"  - {path}: {error}")

    return converted
