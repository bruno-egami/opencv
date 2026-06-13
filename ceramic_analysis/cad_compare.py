# -*- coding: utf-8 -*-
"""
Módulo de Comparação Geométrica com Modelo CAD.
Projeta modelos 3D (STL e STEP) e compara com os contornos das peças extraídos.
"""

import os
import logging
import numpy as np
import cv2
import trimesh
import cadquery as cq
from shapely.geometry import Polygon
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

# Configuração de logging
logger = logging.getLogger("ceramic_analysis.cad_compare")

def _load_step(filepath: str, tolerance: float = 0.05, angular_tolerance: float = 0.1) -> trimesh.Trimesh:
    """
    Converte um arquivo STEP/STP para uma malha trimesh.Trimesh usando CadQuery.
    
    Args:
        filepath: Caminho do arquivo .step ou .stp
        tolerance: Tolerância linear de tesselação em mm
        angular_tolerance: Tolerância angular de tesselação em radianos
        
    Returns:
        trimesh.Trimesh contendo a geometria tessalada.
    """
    logger.info(f"Importando STEP via CadQuery: {filepath}")
    wp = cq.importers.importStep(filepath)
    shape = wp.val()
    if shape is None:
        raise ValueError("Falha ao carregar o modelo STEP: nenhum objeto de geometria encontrado.")
        
    # Realiza a tesselação
    vertices, faces = shape.tessellate(tolerance, angular_tolerance)
    if not vertices or not faces:
        raise ValueError("A tesselação do modelo STEP resultou em uma malha vazia.")
        
    np_vertices = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
    np_faces = np.array(faces, dtype=np.int32)
    
    return trimesh.Trimesh(vertices=np_vertices, faces=np_faces)

def load_cad_model(filepath: str, tolerance: float = 0.05, angular_tolerance: float = 0.1) -> trimesh.Trimesh:
    """
    Carrega o modelo CAD (.stl, .step, .stp).
    Garante que seja retornado um objeto trimesh.Trimesh válido e com normais orientadas.
    
    Args:
        filepath: Caminho do arquivo de modelo CAD
        tolerance: Tolerância linear de tesselação para STEP (mm)
        angular_tolerance: Tolerância angular de tesselação para STEP (rad)
        
    Returns:
        trimesh.Trimesh
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo do modelo CAD não encontrado: {filepath}")
        
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.stl':
        logger.info(f"Carregando STL via Trimesh: {filepath}")
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("O arquivo STL carregado está vazio.")
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    elif ext in ('.step', '.stp'):
        mesh = _load_step(filepath, tolerance, angular_tolerance)
    else:
        raise ValueError(f"Extensão de arquivo CAD não suportada: {ext}. Formatos válidos: .stl, .step, .stp")
        
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("A malha CAD carregada não contém geometria válida.")
        
    # Corrige e orienta as normais da malha para fora
    mesh.fix_normals()
    return mesh

def detect_orientation(mesh: trimesh.Trimesh, symmetry_threshold: float = 0.05) -> dict:
    """
    Detecta a orientação natural do modelo CAD usando PCA (Principal Component Analysis)
    sobre os vértices da malha. Classifica o tipo de geometria e identifica simetrias.
    
    Args:
        mesh: Malha 3D carregada
        symmetry_threshold: Limiar para simetria axial (diferença percentual de autovalores)
        
    Returns:
        Dict contendo:
            rotation_matrix: Matriz de transformação 4x4 para alinhar ao sistema canônico
            extents_mm: Dimensões (largura, profundidade, altura) no sistema alinhado
            is_symmetric: True se a geometria for simétrica em algum plano/eixo
            symmetry_axis: Eixo de simetria (0 para X, 1 para Y, 2 para Z ou None)
            shape_class: Classe da forma ("prismatic", "axisymmetric" ou "organic")
    """
    # Centraliza os vértices temporariamente para o cálculo do PCA
    vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(vertices.T)
    
    # Extrai autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Ordena os componentes de forma decrescente (λ1 >= λ2 >= λ3)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Garante que seja uma matriz de rotação válida (determinante = +1)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]
        
    v1, v2, v3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    
    is_symmetric = False
    symmetry_axis = None
    shape_class = "prismatic"
    
    # Detecção de simetria axial (ex: cilindro ou cone circular ao longo do eixo principal)
    # Se λ2 e λ3 são muito similares, a seção transversal é circular perpendicular a X (λ1)
    if v2 > 0 and abs(v2 - v3) / v2 < symmetry_threshold:
        is_symmetric = True
        symmetry_axis = 0  # Eixo principal X
        shape_class = "axisymmetric"
    # Se λ1 e λ2 são muito similares, seção circular perpendicular a Z (λ3) (ex: disco)
    elif v1 > 0 and abs(v1 - v2) / v1 < symmetry_threshold:
        is_symmetric = True
        symmetry_axis = 2  # Eixo menor Z
        shape_class = "axisymmetric"
        
    # Verifica se os três eixos possuem variâncias muito próximas (orgânico ou esférico)
    if v1 > 0 and abs(v1 - v3) / v1 < 0.20:
        shape_class = "organic"
        
    # Matriz 4x4 de rotação
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = eigenvectors.T
    
    # Para formas orgânicas puras, a rotação do PCA pode oscilar. Usa-se OBB (Oriented Bounding Box)
    if shape_class == "organic":
        logger.info("Forma classificada como orgânica/amorfa. Utilizando OBB para alinhamento inicial estável.")
        to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
        rotation_matrix = to_origin
        
    # Calcula as dimensões (extents) após alinhar a malha
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(rotation_matrix)
    extents_mm = aligned_mesh.bounds[1] - aligned_mesh.bounds[0]
    
    return {
        "rotation_matrix": rotation_matrix,
        "extents_mm": extents_mm,
        "is_symmetric": is_symmetric,
        "symmetry_axis": symmetry_axis,
        "shape_class": shape_class
    }

def align_mesh(mesh: trimesh.Trimesh, orientation: dict) -> trimesh.Trimesh:
    """
    Aplica a transformação de rotação detectada e centraliza a malha na origem (0, 0, 0).
    
    Args:
        mesh: Malha original
        orientation: Dicionário retornado por detect_orientation
        
    Returns:
        Malha alinhada e centralizada
    """
    aligned = mesh.copy()
    aligned.apply_transform(orientation["rotation_matrix"])
    # Centraliza o centroide geométrico exato na origem
    aligned.vertices -= aligned.vertices.mean(axis=0)
    return aligned

def get_view_projection_matrix(view: str) -> np.ndarray:
    """
    Retorna a matriz de orientação 4x4 correspondente para a vista CAD.
    Após a aplicação desta matriz, a projeção ortogonal na câmera de visualização
    corresponde diretamente ao plano XY (descartando-se a coordenada Z).
    
    Vistas:
        top: olhando de +Z para -Z (Plano XY)
        front: olhando de +Y para -Y (Plano XZ)
        back: olhando de -Y para +Y (Plano XZ)
        left: olhando de +X para -X (Plano YZ)
        right: olhando de -X para +X (Plano YZ)
    """
    if view == "top":
        R = np.eye(3)
    elif view == "front":
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    elif view == "back":
        R = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    elif view == "left":
        R = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
    elif view == "right":
        R = np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
    else:
        raise ValueError(f"Vista inválida: {view}")
        
    T = np.eye(4)
    T[:3, :3] = R
    return T

def create_mask_from_polygon(polygon: Polygon, pitch_mm: float = 0.1, padding_px: int = 5) -> tuple[np.ndarray, dict]:
    """
    Cria uma máscara binária (uint8) a partir de um polígono Shapely 2D.
    Útil para representação em formato de imagem e validações de IoU de imagem.
    
    Args:
        polygon: Polígono do contorno
        pitch_mm: Resolução da rasterização (mm/pixel)
        padding_px: Margem de pixels nas bordas
        
    Returns:
        mask: np.ndarray (uint8, 0 e 255)
        transform_info: Dados de mapeamento mm -> pixel
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    w_mm = max_x - min_x
    h_mm = max_y - min_y
    
    w_px = int(np.ceil(w_mm / pitch_mm)) + 2 * padding_px
    h_px = int(np.ceil(h_mm / pitch_mm)) + 2 * padding_px
    
    mask = np.zeros((h_px, w_px), dtype=np.uint8)
    
    # Conversão de mm para coordenada de pixel
    def to_px(coords):
        pts = np.zeros_like(coords, dtype=np.int32)
        pts[:, 0] = np.round((coords[:, 0] - min_x) / pitch_mm + padding_px).astype(np.int32)
        pts[:, 1] = np.round((coords[:, 1] - min_y) / pitch_mm + padding_px).astype(np.int32)
        return pts
        
    # Desenha o exterior preenchido
    ext_px = to_px(np.array(polygon.exterior.coords))
    cv2.fillPoly(mask, [ext_px], 255)
    
    # Desenha os furos como fundo preto (0)
    for hole in polygon.interiors:
        hole_px = to_px(np.array(hole.coords))
        cv2.fillPoly(mask, [hole_px], 0)
        
    transform_info = {
        "min_x": min_x,
        "min_y": min_y,
        "pitch_mm": pitch_mm,
        "padding_px": padding_px
    }
    
    return mask, transform_info

def project_to_2d(
    mesh: trimesh.Trimesh,
    view: str = "top",
    pitch_mm: float = 0.1,
) -> tuple[np.ndarray, list, np.ndarray, dict]:
    """
    Projeta ortogonalmente a malha 3D em um contorno 2D para a vista especificada.
    Trata múltiplos polígonos e furos em peças vazadas.
    
    Args:
        mesh: Malha 3D alinhada e centralizada
        view: Nome da vista
        pitch_mm: Resolução espacial para a máscara rasterizada
        
    Returns:
        exterior_contour_mm: Contorno externo em milímetros (Nx2)
        hole_contours_mm: Lista de contornos de furos em milímetros (M x [Px2])
        mask: Máscara binária rasterizada da projeção (uint8)
        bbox_mm: Metadados da bounding box {width_mm, height_mm, area_mm2}
    """
    # Rotaciona para alinhar a direção de visão da câmera ao plano XY
    T = get_view_projection_matrix(view)
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(T)
    
    # Projeta no plano XY (normal [0, 0, 1])
    proj = rotated_mesh.projected(normal=[0, 0, 1])
    
    if len(proj.polygons_full) == 0:
        raise ValueError(f"A projeção 2D da malha na vista '{view}' resultou em um objeto vazio.")
        
    # Ordena os polígonos projetados por área decrescente (o maior representa a silhueta principal)
    polygons = sorted(proj.polygons_full, key=lambda p: p.area, reverse=True)
    polygon = polygons[0]
    
    # Extrai o contorno externo (removendo o ponto duplicado no final que o Shapely gera)
    exterior_contour_mm = np.array(polygon.exterior.coords, dtype=np.float64)
    if len(exterior_contour_mm) > 1 and np.allclose(exterior_contour_mm[0], exterior_contour_mm[-1]):
        exterior_contour_mm = exterior_contour_mm[:-1]
        
    # Extrai os furos internos (interiors)
    hole_contours_mm = []
    for interior in polygon.interiors:
        hole_coords = np.array(interior.coords, dtype=np.float64)
        if len(hole_coords) > 1 and np.allclose(hole_coords[0], hole_coords[-1]):
            hole_coords = hole_coords[:-1]
        hole_contours_mm.append(hole_coords)
        
    # Cria máscara binária
    mask, _ = create_mask_from_polygon(polygon, pitch_mm)
    
    # Informações de dimensão
    min_x, min_y, max_x, max_y = polygon.bounds
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    area_mm2 = polygon.area
    
    bbox_mm = {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "area_mm2": area_mm2
    }
    
    return exterior_contour_mm, hole_contours_mm, mask, bbox_mm

def resample_contour(
    contour_mm: np.ndarray,
    target_spacing_mm: float = 0.5
) -> np.ndarray:
    """
    Reamostra adaptativamente um contorno fechado 2D com base no comprimento de arco
    para garantir que os pontos fiquem distribuídos de forma homogênea.
    
    Args:
        contour_mm: Contorno original (Nx2, mm)
        target_spacing_mm: Distância desejada entre pontos consecutivos
        
    Returns:
        Contorno reamostrado (Mx2, mm)
    """
    if len(contour_mm) < 3:
        return contour_mm.copy()
        
    # Garante que o contorno seja fechado para calcular o comprimento completo
    pts = np.vstack([contour_mm, contour_mm[0]])
    
    # Comprimento de arco cumulativo
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cumulative_length = np.insert(np.cumsum(dists), 0, 0.0)
    total_length = cumulative_length[-1]
    
    if total_length < 1e-6:
        return contour_mm.copy()
        
    # Número de pontos uniformemente espaçados
    num_points = int(np.round(total_length / target_spacing_mm))
    num_points = max(num_points, 10)  # Pelo menos 10 pontos
    
    # Espaçamento uniforme (descartando o último ponto que coincide com o primeiro)
    new_lengths = np.linspace(0.0, total_length, num_points + 1)[:-1]
    
    # Interpolação linear das coordenadas
    new_x = np.interp(new_lengths, cumulative_length, pts[:, 0])
    new_y = np.interp(new_lengths, cumulative_length, pts[:, 1])
    
    return np.column_stack([new_x, new_y])

def compute_shape_complexity(contour: np.ndarray) -> float:
    """
    Calcula a complexidade geométrica do contorno como (perímetro² / área).
    Círculo perfeito ≈ 12.57, Quadrado = 16. Formas complexas > 20.
    
    Args:
        contour: Contorno em mm (Nx2)
        
    Returns:
        Valor da complexidade
    """
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    area = poly.area
    perimeter = poly.length
    
    if area < 1e-6:
        return 0.0
        
    return (perimeter ** 2) / area

def _icp_2d(
    source: np.ndarray,
    target: np.ndarray,
    max_iter: int = 50,
    tolerance: float = 0.01,
    allow_rotation: bool = True
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Iterative Closest Point (ICP) em 2D.
    Ajusta a fonte (source) ao destino (target) minimizando os desvios.
    
    Args:
        source: Contorno móvel (CAD)
        target: Contorno fixo (Foto)
        max_iter: Máximo de iterações
        tolerance: Limiar de parada por variação média
        allow_rotation: Permite rotacionar (desativar para peças perfeitamente simétricas)
        
    Returns:
        src: Contorno transformado (Nx2)
        mean_error: Erro médio final obtido
        T_cum: Matriz de transformação afim 3x3 cumulativa
    """
    src = source.copy()
    prev_error = float('inf')
    T_cum = np.eye(3)
    
    # Árvore KD para busca rápida de correspondências
    tree = KDTree(target)
    
    for i in range(max_iter):
        distances, indices = tree.query(src)
        target_matched = target[indices]
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
        # Centroides dos pontos correspondentes
        c_src = np.mean(src, axis=0)
        c_tgt = np.mean(target_matched, axis=0)
        
        src_centered = src - c_src
        tgt_centered = target_matched - c_tgt
        
        # Calcula R e t via SVD
        if allow_rotation:
            H = np.dot(src_centered.T, tgt_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # Corrige reflexão indesejada
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = np.dot(Vt.T, U.T)
        else:
            R = np.eye(2)
            
        t = c_tgt - np.dot(R, c_src)
        
        # Aplica a transformação do passo aos pontos
        src = np.dot(src, R.T) + t
        
        # Acumula matriz de transformação
        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[:2, 2] = t
        T_cum = np.dot(T_step, T_cum)
        
    return src, prev_error, T_cum

def apply_registration(points_mm: np.ndarray, transform: dict) -> np.ndarray:
    """
    Aplica uma transformação geométrica 2D (derivada do registro) a um conjunto de pontos.
    
    Args:
        points_mm: Array de pontos (Nx2)
        transform: Dicionário contendo os dados do registro (rotação e translação)
        
    Returns:
        Pontos transformados (Nx2)
    """
    method = transform.get("method_used", "")
    
    if "icp" in method:
        rot_deg = transform["rotation_deg"]
        rad = np.radians(rot_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        t = transform["translation_mm"]
        return np.dot(points_mm, R.T) + t
    else:
        # centroid ou bbox_center (apenas translação direta)
        return points_mm + transform["translation_mm"]

def register_contours(
    cad_contour_mm: np.ndarray,
    photo_contour_mm: np.ndarray,
    method: str = "icp",
    shape_class: str = "prismatic"
) -> tuple[np.ndarray, dict]:
    """
    Registra (alinha) o contorno do CAD com o contorno da foto física.
    Suporta alinhamento por centroide, bounding box ou ICP 2D com e sem rotação.
    
    Args:
        cad_contour_mm: Contorno CAD projetado (mm)
        photo_contour_mm: Contorno extraído da foto (mm)
        method: Método a utilizar ("icp", "centroid" ou "bbox_center")
        shape_class: Classe da forma ("prismatic", "axisymmetric" ou "organic")
        
    Returns:
        cad_aligned: Contorno CAD transformado/alinhado (Nx2)
        transform_info: Dicionário com dados do alinhamento executado
    """
    c_cad = np.mean(cad_contour_mm, axis=0)
    c_photo = np.mean(photo_contour_mm, axis=0)
    
    # Inicializa o alinhamento de translação de centroides
    translation_init = c_photo - c_cad
    cad_centered = cad_contour_mm + translation_init
    
    if method == "centroid":
        # Apenas alinhamento de centroides
        dists = np.sqrt(np.sum((cad_centered - photo_contour_mm[KDTree(photo_contour_mm).query(cad_centered)[1]])**2, axis=1))
        return cad_centered, {
            "rotation_deg": 0.0,
            "translation_mm": translation_init,
            "rms_error_mm": np.mean(dists),
            "method_used": "centroid"
        }
        
    elif method == "bbox_center":
        # Alinhamento pelo centro das Bounding Boxes
        min_cad, max_cad = np.min(cad_contour_mm, axis=0), np.max(cad_contour_mm, axis=0)
        min_pho, max_pho = np.min(photo_contour_mm, axis=0), np.max(photo_contour_mm, axis=0)
        bc_cad = (min_cad + max_cad) / 2.0
        bc_pho = (min_pho + max_pho) / 2.0
        translation_bbox = bc_pho - bc_cad
        cad_aligned = cad_contour_mm + translation_bbox
        dists = np.sqrt(np.sum((cad_aligned - photo_contour_mm[KDTree(photo_contour_mm).query(cad_aligned)[1]])**2, axis=1))
        return cad_aligned, {
            "rotation_deg": 0.0,
            "translation_mm": translation_bbox,
            "rms_error_mm": np.mean(dists),
            "method_used": "bbox_center"
        }
        
    # Método ICP
    # Se a geometria é axisymmetric (ex: cilindros e cones circulares na vista top),
    # a rotação livre na projeção gera múltiplos equivalentes. Bloqueia-se a rotação.
    allow_rotation = (shape_class != "axisymmetric")
    
    if not allow_rotation:
        logger.info("Registro ICP: Geometria axissimétrica detectada. Registro limitado a translação apenas.")
        aligned_pts, rms_error, T = _icp_2d(
            cad_contour_mm, photo_contour_mm,
            max_iter=50, tolerance=0.01, allow_rotation=False
        )
        return aligned_pts, {
            "rotation_deg": 0.0,
            "translation_mm": T[:2, 2],
            "rms_error_mm": rms_error,
            "method_used": "icp_translation_only"
        }
        
    # Para formas prismáticas ou orgânicas, testa orientações iniciais (0, 90, 180, 270)
    # para evitar que o ICP convirja para mínimos locais.
    best_pts = None
    best_rms = float('inf')
    best_T = None
    
    # Move para a origem para testar rotações iniciais limpas
    cad_temp = cad_contour_mm - c_cad
    
    angles_deg = [0, 90, 180, 270] if shape_class == "prismatic" else [0]
    
    for angle in angles_deg:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        R_init = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotaciona e translada para o centroide da foto
        src_init = np.dot(cad_temp, R_init.T) + c_photo
        
        max_it = 100 if shape_class == "organic" else 50
        aligned_pts, rms_error, T = _icp_2d(
            src_init, photo_contour_mm,
            max_iter=max_it, tolerance=0.01, allow_rotation=True
        )
        
        if rms_error < best_rms:
            best_rms = rms_error
            best_pts = aligned_pts
            # Calcula a matriz global combinada
            T_init = np.eye(3)
            T_init[:2, :2] = R_init
            T_init[:2, 2] = c_photo - np.dot(R_init, c_cad)
            best_T = np.dot(T, T_init)
            
    R_final = best_T[:2, :2]
    rot_rad = np.arctan2(R_final[1, 0], R_final[0, 0])
    rot_deg = np.degrees(rot_rad)
    trans = best_T[:2, 2]
    
    return best_pts, {
        "rotation_deg": rot_deg,
        "translation_mm": trans,
        "rms_error_mm": best_rms,
        "method_used": "icp"
    }

def fit_circle(contour: np.ndarray) -> tuple[float, float, float]:
    """
    Ajusta um círculo geométrico ao contorno usando estimativa de centroide.
    Gera o centro (xc, yc) e o diâmetro da circunferência.
    """
    xc, yc = np.mean(contour, axis=0)
    r = np.mean(np.sqrt((contour[:, 0] - xc)**2 + (contour[:, 1] - yc)**2))
    return xc, yc, 2 * r

def compare_contours(
    cad_contour_mm: np.ndarray,
    photo_contour_mm: np.ndarray,
    cad_bbox: dict,
    cad_holes: list = None,
    photo_holes: list = None,
    shape_class: str = "prismatic"
) -> dict:
    """
    Calcula desvios dimensionais e métricas de fidelidade geométrica (Hausdorff, IoU, desvios).
    
    Args:
        cad_contour_mm: Contorno CAD alinhado (mm)
        photo_contour_mm: Contorno da foto (mm)
        cad_bbox: Dados do bounding box do CAD
        cad_holes: Furos do CAD
        photo_holes: Furos da foto
        shape_class: Classe da forma
        
    Returns:
        Dicionário com todas as métricas calculadas
    """
    # 1. Distância de Hausdorff bidirecional simétrica
    h_ab = directed_hausdorff(cad_contour_mm, photo_contour_mm)[0]
    h_ba = directed_hausdorff(photo_contour_mm, cad_contour_mm)[0]
    hausdorff_mm = max(h_ab, h_ba)
    
    # 2. Desvio estatístico ponto a ponto (CAD -> Foto mais próximo)
    tree_photo = KDTree(photo_contour_mm)
    distances, _ = tree_photo.query(cad_contour_mm)
    mean_deviation_mm = np.mean(distances)
    deviation_std_mm = np.std(distances)
    deviation_p95_mm = np.percentile(distances, 95)
    
    # 3. IoU analítico via Shapely
    cad_poly = Polygon(cad_contour_mm)
    photo_poly = Polygon(photo_contour_mm)
    
    if not cad_poly.is_valid:
        cad_poly = cad_poly.buffer(0)
    if not photo_poly.is_valid:
        photo_poly = photo_poly.buffer(0)
        
    inter = cad_poly.intersection(photo_poly).area
    union = cad_poly.union(photo_poly).area
    iou = inter / union if union > 0 else 0.0
    
    # 4. Desvios de Bounding Box e Área
    min_pho, max_pho = np.min(photo_contour_mm, axis=0), np.max(photo_contour_mm, axis=0)
    photo_w = max_pho[0] - min_pho[0]
    photo_h = max_pho[1] - min_pho[1]
    photo_area = photo_poly.area
    
    bbox_w_deviation_mm = photo_w - cad_bbox["width_mm"]
    bbox_h_deviation_mm = photo_h - cad_bbox["height_mm"]
    
    bbox_w_deviation_pct = (bbox_w_deviation_mm / cad_bbox["width_mm"]) * 100.0 if cad_bbox["width_mm"] > 0 else 0.0
    bbox_h_deviation_pct = (bbox_h_deviation_mm / cad_bbox["height_mm"]) * 100.0 if cad_bbox["height_mm"] > 0 else 0.0
    area_deviation_pct = ((photo_area - cad_bbox["area_mm2"]) / cad_bbox["area_mm2"]) * 100.0 if cad_bbox["area_mm2"] > 0 else 0.0
    
    complexity = compute_shape_complexity(cad_contour_mm)
    
    results = {
        "cad_bbox_w_mm": cad_bbox["width_mm"],
        "cad_bbox_h_mm": cad_bbox["height_mm"],
        "cad_area_mm2": cad_bbox["area_mm2"],
        "measured_bbox_w_mm": photo_w,
        "measured_bbox_h_mm": photo_h,
        "measured_area_mm2": photo_area,
        "bbox_w_deviation_mm": bbox_w_deviation_mm,
        "bbox_h_deviation_mm": bbox_h_deviation_mm,
        "bbox_w_deviation_pct": bbox_w_deviation_pct,
        "bbox_h_deviation_pct": bbox_h_deviation_pct,
        "area_deviation_pct": area_deviation_pct,
        "hausdorff_mm": hausdorff_mm,
        "mean_deviation_mm": mean_deviation_mm,
        "deviation_std_mm": deviation_std_mm,
        "deviation_p95_mm": deviation_p95_mm,
        "iou": iou,
        "shape_complexity": complexity,
        "per_point_distances_mm": distances
    }
    
    # 5. Métricas de Axissimétricos (diâmetro/concentricidade)
    # Classificamos por proximidade circular ou shape_class
    photo_circularity = (4 * np.pi * photo_area) / (photo_poly.length ** 2) if photo_poly.length > 0 else 0
    if shape_class == "axisymmetric" or photo_circularity > 0.85:
        xc_c, yc_c, d_cad = fit_circle(cad_contour_mm)
        xc_p, yc_p, d_pho = fit_circle(photo_contour_mm)
        
        diameter_deviation_mm = d_pho - d_cad
        diameter_deviation_pct = (diameter_deviation_mm / d_cad) * 100.0 if d_cad > 0 else 0.0
        concentricity_mm = np.sqrt((xc_c - xc_p)**2 + (yc_c - yc_p)**2)
        
        results.update({
            "diameter_cad_mm": d_cad,
            "diameter_photo_mm": d_pho,
            "diameter_deviation_mm": diameter_deviation_mm,
            "diameter_deviation_pct": diameter_deviation_pct,
            "concentricity_mm": concentricity_mm
        })
        
    # 6. Métricas de Orgânicas/Vazadas com Furos
    n_holes_cad = len(cad_holes) if cad_holes else 0
    n_holes_photo = len(photo_holes) if photo_holes else 0
    
    results.update({
        "n_holes_cad": n_holes_cad,
        "n_holes_photo": n_holes_photo,
        "holes_matched": (n_holes_cad == n_holes_photo)
    })
    
    if n_holes_cad > 0 or n_holes_photo > 0:
        cad_poly_holes = Polygon(cad_contour_mm, cad_holes if cad_holes else [])
        photo_poly_holes = Polygon(photo_contour_mm, photo_holes if photo_holes else [])
        
        if not cad_poly_holes.is_valid:
            cad_poly_holes = cad_poly_holes.buffer(0)
        if not photo_poly_holes.is_valid:
            photo_poly_holes = photo_poly_holes.buffer(0)
            
        inter_h = cad_poly_holes.intersection(photo_poly_holes).area
        union_h = cad_poly_holes.union(photo_poly_holes).area
        holes_iou = inter_h / union_h if union_h > 0 else 0.0
        results["holes_iou"] = holes_iou
    else:
        results["holes_iou"] = iou
        
    return results

def generate_deviation_map(
    photo_image: np.ndarray,
    cad_contour_px: np.ndarray,
    photo_contour_px: np.ndarray,
    per_point_distances_mm: np.ndarray,
    output_path: str,
    px_per_mm_h: float,
    px_per_mm_v: float,
    tolerance_mm: float = 1.0,
    cad_holes_px: list = None,
    metrics: dict = None
) -> np.ndarray:
    """
    Gera uma imagem de mapa de desvios visualmente premium.
    O contorno CAD é plotado como um conjunto de pontos coloridos representando o desvio local.
    O contorno da foto física é desenhado em Ciano.
    
    Args:
        photo_image: Imagem original (física) da peça
        cad_contour_px: Contorno CAD alinhado em coordenadas de pixels (Nx2)
        photo_contour_px: Contorno extraído da foto física em pixels (Mx2)
        per_point_distances_mm: Array de desvios (mm) em cada ponto do CAD
        output_path: Caminho para salvar a imagem gerada
        px_per_mm_h: Fator de escala horizontal
        px_per_mm_v: Fator de escala vertical
        tolerance_mm: Limiar de tolerância geométrico (mm)
        cad_holes_px: Lista de contornos de furos do CAD em pixels
        metrics: Dicionário com as métricas agregadas para plotagem da legenda
        
    Returns:
        Imagem anotada (np.ndarray)
    """
    # Garante imagem em formato BGR colorido
    if len(photo_image.shape) == 2:
        annotated = cv2.cvtColor(photo_image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = photo_image.copy()
        
    # 1. Desenha a legenda de métricas com fundo semi-transparente
    overlay = annotated.copy()
    h_img, w_img = annotated.shape[:2]
    
    text_w = 360
    text_h = 240 if metrics else 100
    # Retângulo de fundo para o texto
    cv2.rectangle(overlay, (10, 10), (10 + text_w, 10 + text_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
    
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_white = (255, 255, 255)
    
    def put_text(text, color=color_white):
        nonlocal y_offset
        cv2.putText(annotated, text, (20, y_offset), font, font_scale, color, 1, cv2.LINE_AA)
        y_offset += 20
        
    put_text("COMPARAÇÃO COM MODELO CAD", (0, 255, 255))
    put_text(f"Tolerancia Limite: {tolerance_mm:.2f} mm")
    
    if metrics:
        put_text(f"Hausdorff Max: {metrics['hausdorff_mm']:.3f} mm")
        put_text(f"Desvio Medio: {metrics['mean_deviation_mm']:.3f} mm (std: {metrics['deviation_std_mm']:.3f})")
        put_text(f"IoU Alinhamento: {metrics['iou']:.3f}")
        put_text(f"Complexidade da Forma: {metrics['shape_complexity']:.2f}")
        
        if "diameter_photo_mm" in metrics:
            put_text(f"Dia. Medido/CAD: {metrics['diameter_photo_mm']:.2f}/{metrics['diameter_cad_mm']:.2f} mm")
            put_text(f"Desvio Dia.: {metrics['diameter_deviation_mm']:.3f} mm ({metrics['diameter_deviation_pct']:.2f}%)")
            put_text(f"Concentricidade: {metrics['concentricity_mm']:.3f} mm")
        else:
            put_text(f"Desvio Largura (W): {metrics['bbox_w_deviation_mm']:.3f} mm ({metrics['bbox_w_deviation_pct']:.2f}%)")
            put_text(f"Desvio Altura (H): {metrics['bbox_h_deviation_mm']:.3f} mm ({metrics['bbox_h_deviation_pct']:.2f}%)")
            put_text(f"Desvio de Area: {metrics['area_deviation_pct']:.2f}%")
            
        if "n_holes_photo" in metrics and metrics["n_holes_cad"] > 0:
            put_text(f"Furos (Foto/CAD): {metrics['n_holes_photo']}/{metrics['n_holes_cad']} (IoU: {metrics.get('holes_iou', 0.0):.3f})")
            
    # 2. Desenha o contorno da foto em Ciano (sólido)
    cv2.polylines(annotated, [photo_contour_px.astype(np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
    
    # 3. Desenha os furos do CAD (se houverem) em Cinza tracejado/sólido
    if cad_holes_px:
        for hole in cad_holes_px:
            cv2.polylines(annotated, [hole.astype(np.int32)], isClosed=True, color=(128, 128, 128), thickness=1)
            
    # 4. Desenha os pontos do contorno CAD como esferas coloridas baseadas no desvio local
    for i, p in enumerate(cad_contour_px):
        d = per_point_distances_mm[i]
        
        # Mapeamento do mapa de calor de desvio
        if d < tolerance_mm * 0.5:
            color = (0, 255, 0)      # Verde (dentro da tolerância ideal)
        elif d < tolerance_mm:
            color = (0, 255, 255)    # Amarelo (alerta de desvio moderado)
        else:
            color = (0, 0, 255)      # Vermelho (fora da tolerância permitida)
            
        cv2.circle(annotated, (int(round(p[0])), int(round(p[1]))), 3, color, -1, cv2.LINE_AA)
        
    # 5. Desenha a barra escala de cores de desvio no canto inferior esquerdo
    scale_y = h_img - 80
    cv2.rectangle(annotated, (10, scale_y), (30, scale_y + 15), (0, 255, 0), -1)
    cv2.putText(annotated, f"< {tolerance_mm * 0.5:.2f} mm (Ideal)", (40, scale_y + 12), font, 0.45, color_white, 1, cv2.LINE_AA)
    
    scale_y += 20
    cv2.rectangle(annotated, (10, scale_y), (30, scale_y + 15), (0, 255, 255), -1)
    cv2.putText(annotated, f"{tolerance_mm * 0.5:.2f} - {tolerance_mm:.2f} mm (Alerta)", (40, scale_y + 12), font, 0.45, color_white, 1, cv2.LINE_AA)
    
    scale_y += 20
    cv2.rectangle(annotated, (10, scale_y), (30, scale_y + 15), (0, 0, 255), -1)
    cv2.putText(annotated, f"> {tolerance_mm:.2f} mm (Critico)", (40, scale_y + 12), font, 0.45, color_white, 1, cv2.LINE_AA)
    
    # Salva a imagem
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    logger.info(f"Mapa de desvio visual salvo em: {output_path}")
    
    return annotated
