# -*- coding: utf-8 -*-
"""
Módulo interativo para ajuste manual do contorno das peças via interface gráfica OpenCV.
"""

import cv2
import numpy as np
import logging
import ctypes
import config

logger = logging.getLogger("ceramic_analysis.interactive")


class InteractiveContourEditor:
    def __init__(self, image, auto_contour, window_title="Ajuste de Contorno"):
        self.img = image.copy()
        self.window_title = window_title
        
        # Guardar contorno original para referência
        self.original_contour = auto_contour.copy()
        
        # Inicializar vértices a partir do retângulo de área mínima (4 cantos)
        rect = cv2.minAreaRect(auto_contour)
        box = cv2.boxPoints(rect)
        self.vertices = [list(pt) for pt in box]
        self.initial_vertices = [list(pt) for pt in box]
        
        # Dimensões da janela ajustadas pela resolução e aspect ratio
        img_h, img_w = self.img.shape[:2]
        self.window_width = config.INTERACTIVE_WINDOW_WIDTH
        aspect = img_h / img_w
        self.window_height = int(self.window_width * aspect)
        if self.window_height > 900:
            self.window_height = 900
            self.window_width = int(self.window_height / aspect)
            
        # Parâmetros de Zoom e Pan (câmera)
        scale_w = self.window_width / img_w
        scale_h = self.window_height / img_h
        self.s = min(scale_w, scale_h) * 0.95
        self.tx = (self.window_width - img_w * self.s) / 2
        self.ty = (self.window_height - img_h * self.s) / 2
        
        # Salvar padrões para o reset
        self.default_s = self.s
        self.default_tx = self.tx
        self.default_ty = self.ty
        
        # Configurações visuais e snap
        self.vertex_radius = config.INTERACTIVE_VERTEX_RADIUS
        self.snap_distance = config.INTERACTIVE_SNAP_DISTANCE
        
        # Estados de interação
        self.hovered_idx = None
        self.selected_idx = None
        self.is_dragging = False
        self.is_panning = False
        
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_tx = 0.0
        self.pan_start_ty = 0.0
        
        # Flags de controle
        self.confirmed = False
        self.was_adjusted = False
        
    def to_screen(self, pt):
        """Converte coordenadas da imagem para coordenadas da tela (janela)."""
        x_s = self.s * pt[0] + self.tx
        y_s = self.s * pt[1] + self.ty
        return int(round(x_s)), int(round(y_s))

    def to_image(self, screen_pt):
        """Converte coordenadas da tela (janela) para coordenadas da imagem original."""
        x_i = (screen_pt[0] - self.tx) / self.s
        y_i = (screen_pt[1] - self.ty) / self.s
        return x_i, y_i

    def mouse_callback(self, event, x, y, flags, param):
        """Manipulador de eventos do mouse do OpenCV."""
        # Obter coordenadas no espaço da imagem
        ix, iy = self.to_image((x, y))
        
        # 1. Atualizar índice do vértice sob o mouse (hover)
        self.hovered_idx = None
        min_dist = float('inf')
        for i, pt in enumerate(self.vertices):
            sx, sy = self.to_screen(pt)
            dist = np.hypot(x - sx, y - sy)
            if dist < self.snap_distance and dist < min_dist:
                min_dist = dist
                self.hovered_idx = i
                
        # 2. Scroll do mouse (Zoom centrado no cursor)
        if event == cv2.EVENT_MOUSEWHEEL:
            signed_flags = ctypes.c_int32(flags).value
            zoom_factor = 1.15 if signed_flags > 0 else (1.0 / 1.15)
            
            new_s = self.s * zoom_factor
            # Limitar escala
            if 0.05 <= new_s <= 100.0:
                self.s = new_s
                self.tx = x - self.s * ix
                self.ty = y - self.s * iy
                
        # 3. Clique do Botão Esquerdo (Selecionar/Iniciar arraste de vértice)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.hovered_idx is not None:
                self.selected_idx = self.hovered_idx
                self.is_dragging = True
            else:
                self.selected_idx = None
                
        # 4. Movimento do Mouse (Arrastar vértice ou Pan)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.selected_idx is not None:
                img_h, img_w = self.img.shape[:2]
                # Limitar vértice aos limites da imagem
                ix_clamped = max(0.0, min(float(img_w - 1), ix))
                iy_clamped = max(0.0, min(float(img_h - 1), iy))
                self.vertices[self.selected_idx] = [ix_clamped, iy_clamped]
                self.was_adjusted = True
            elif self.is_panning:
                self.tx = self.pan_start_tx + (x - self.pan_start_x)
                self.ty = self.pan_start_ty + (y - self.pan_start_y)
                
        # 5. Soltar Botão Esquerdo
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            
        # 6. Clique do Botão Direito (Iniciar Pan)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            self.pan_start_tx = self.tx
            self.pan_start_ty = self.ty
            
        # 7. Soltar Botão Direito
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            
        # 8. Duplo-clique Esquerdo (Adicionar novo vértice na aresta mais próxima)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            num_v = len(self.vertices)
            if num_v < 100:
                best_idx = -1
                best_dist = float('inf')
                best_proj = None
                
                for i in range(num_v):
                    p1 = np.array(self.vertices[i])
                    p2 = np.array(self.vertices[(i + 1) % num_v])
                    p = np.array([ix, iy])
                    
                    v = p2 - p1
                    w = p - p1
                    v_len_sq = np.dot(v, v)
                    if v_len_sq == 0:
                        t = 0.0
                    else:
                        t = np.dot(w, v) / v_len_sq
                    t = max(0.0, min(1.0, t))
                    projection = p1 + t * v
                    dist = np.linalg.norm(p - projection)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_proj = projection
                
                best_dist_screen = best_dist * self.s
                # Somente adiciona se estiver a menos de 50px de distância da aresta na tela
                if best_idx != -1 and best_dist_screen < 50.0:
                    self.vertices.insert(best_idx + 1, list(best_proj))
                    self.selected_idx = best_idx + 1
                    self.was_adjusted = True

    def run(self):
        """Loop de exibição e captura de teclas."""
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.window_width, self.window_height)
        cv2.setMouseCallback(
            self.window_title,
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param)
        )
        
        while True:
            # 1. Gerar imagem da câmera (Zoom e Pan)
            M = np.float32([[self.s, 0, self.tx], [0, self.s, self.ty]])
            view = cv2.warpAffine(
                self.img, M, (self.window_width, self.window_height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30)
            )
            
            # 2. Desenhar contorno automático original como referência sutil (vermelho)
            pts_original_screen = []
            for pt in self.original_contour:
                x_i, y_i = pt[0]
                pts_original_screen.append(self.to_screen((x_i, y_i)))
            
            if len(pts_original_screen) > 0:
                pts_original_screen = np.array(pts_original_screen, dtype=np.int32).reshape((-1, 1, 2))
                overlay = view.copy()
                cv2.polylines(overlay, [pts_original_screen], isClosed=True, color=(100, 100, 240), thickness=2, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.4, view, 0.6, 0, view)
            
            # 3. Desenhar arestas do contorno ajustável (verde)
            num_v = len(self.vertices)
            screen_pts = []
            for pt in self.vertices:
                screen_pts.append(self.to_screen(pt))
                
            for i in range(num_v):
                pt1 = screen_pts[i]
                pt2 = screen_pts[(i + 1) % num_v]
                cv2.line(view, pt1, pt2, (80, 220, 80), 2, cv2.LINE_AA)
                
            # 4. Desenhar vértices arrastáveis
            for i, pt in enumerate(screen_pts):
                if i == self.selected_idx:
                    color = (50, 100, 255)   # Laranja/Vermelho (Selecionado/Arrastando)
                    radius = self.vertex_radius + 4
                elif i == self.hovered_idx:
                    color = (50, 255, 255)   # Amarelo/Verde (Hover)
                    radius = self.vertex_radius + 2
                else:
                    color = (240, 150, 50)   # Azul Claro (Normal)
                    radius = self.vertex_radius
                    
                cv2.circle(view, pt, radius, color, -1, cv2.LINE_AA)
                cv2.circle(view, pt, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                
            # 5. Desenhar painel translúcido de instruções no topo
            overlay_help = view.copy()
            cv2.rectangle(overlay_help, (10, 10), (self.window_width - 10, 80), (15, 15, 15), -1)
            cv2.addWeighted(overlay_help, 0.75, view, 0.25, 0, view)
            
            instructions = [
                "Arrastar Ponto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Mover: Click direito + arrastar",
                "Adicionar Ponto: Duplo-click na linha       |  Remover Ponto: Selecionar + [Delete]/[D]",
                "Confirmar: [Enter] / [Espaço]               |  Resetar: [R]  |  Cancelar (Auto): [Esc] / [Q]"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(view, text, (20, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
                
            # Exibir imagem
            cv2.imshow(self.window_title, view)
            
            # Aguardar evento de teclado
            key = cv2.waitKey(15)
            if key == -1:
                continue
                
            val = key & 0xFF
            
            # Enter (13) ou Espaço (32) para Confirmar
            if val in [13, 32]:
                self.confirmed = True
                break
                
            # Esc (27) ou 'q'/'Q' para Cancelar
            elif val in [27, ord('q'), ord('Q')]:
                self.confirmed = False
                break
                
            # 'r'/'R' para Resetar
            elif val in [ord('r'), ord('R')]:
                self.vertices = [list(pt) for pt in self.initial_vertices]
                self.selected_idx = None
                self.s = self.default_s
                self.tx = self.default_tx
                self.ty = self.default_ty
                self.was_adjusted = False
                
            # Delete ou 'd'/'D' ou Backspace para deletar ponto selecionado
            elif val in [ord('d'), ord('D')] or key in [3014656, 65535, 46, 2424832, 127]:
                if len(self.vertices) > 3 and self.selected_idx is not None:
                    self.vertices.pop(self.selected_idx)
                    self.selected_idx = None
                    self.was_adjusted = True
                    
        cv2.destroyWindow(self.window_title)
        
        if self.confirmed and self.was_adjusted:
            # Converter de volta para formato de contorno OpenCV (Nx1x2, np.int32)
            adjusted_contour = np.array(np.round(self.vertices), dtype=np.int32).reshape((-1, 1, 2))
            return adjusted_contour, True
        else:
            return self.original_contour, False


# Caches em memória persistentes durante a execução do processo
_ADJUSTED_GRID_CACHE = {}
_MANUAL_CORNERS_CACHE = {}


def adjust_contour(image, auto_contour, window_title="Ajuste de Contorno"):
    """
    Interface pública para acionar a edição interativa de contorno.
    
    Args:
        image: Imagem original colorida (BGR, uint8)
        auto_contour: Contorno OpenCV original detectado (Nx1x2)
        window_title: Título da janela OpenCV
        
    Returns:
        tuple: (contour_ajustado, foi_ajustado_bool)
    """
    try:
        editor = InteractiveContourEditor(image, auto_contour, window_title)
        return editor.run()
    except Exception as e:
        logger.error(f"Erro no ajuste manual de contorno: {e}. Mantendo contorno original.")
        return auto_contour, False


# ══════════════════════════════════════════════════════════════════════════════
# AJUSTE DA GRADE DE CALIBRAÇÃO (PONTOS DE INTERSEÇÃO)
# ══════════════════════════════════════════════════════════════════════════════

class InteractiveGridEditor:
    def __init__(self, image, auto_points, window_title="Ajuste da Grade de Calibracao"):
        self.img = image.copy()
        self.window_title = window_title
        
        # Guardar pontos originais para referência
        if auto_points is not None and len(auto_points) > 0:
            self.original_points = auto_points.copy()
        else:
            self.original_points = np.array([], dtype=np.float32).reshape((0, 2))
            
        self.vertices = [list(pt) for pt in self.original_points]
        self.initial_vertices = [list(pt) for pt in self.original_points]
        
        # Dimensões da janela ajustadas
        img_h, img_w = self.img.shape[:2]
        self.window_width = config.INTERACTIVE_WINDOW_WIDTH
        aspect = img_h / img_w
        self.window_height = int(self.window_width * aspect)
        if self.window_height > 900:
            self.window_height = 900
            self.window_width = int(self.window_height / aspect)
            
        # Parâmetros de Zoom e Pan
        scale_w = self.window_width / img_w
        scale_h = self.window_height / img_h
        self.s = min(scale_w, scale_h) * 0.95
        self.tx = (self.window_width - img_w * self.s) / 2
        self.ty = (self.window_height - img_h * self.s) / 2
        
        self.default_s = self.s
        self.default_tx = self.tx
        self.default_ty = self.ty
        
        # Configurações visuais (menor para grade densa)
        self.vertex_radius = 4
        self.snap_distance = 12
        
        # Estados
        self.hovered_idx = None
        self.selected_idx = None
        self.is_dragging = False
        self.is_panning = False
        
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_tx = 0.0
        self.pan_start_ty = 0.0
        
        self.confirmed = False
        self.was_adjusted = False
        
    def to_screen(self, pt):
        x_s = self.s * pt[0] + self.tx
        y_s = self.s * pt[1] + self.ty
        return int(round(x_s)), int(round(y_s))

    def to_image(self, screen_pt):
        x_i = (screen_pt[0] - self.tx) / self.s
        y_i = (screen_pt[1] - self.ty) / self.s
        return x_i, y_i

    def mouse_callback(self, event, x, y, flags, param):
        ix, iy = self.to_image((x, y))
        
        # 1. Hover
        self.hovered_idx = None
        min_dist = float('inf')
        for i, pt in enumerate(self.vertices):
            sx, sy = self.to_screen(pt)
            dist = np.hypot(x - sx, y - sy)
            if dist < self.snap_distance and dist < min_dist:
                min_dist = dist
                self.hovered_idx = i
                
        # 2. Zoom via Wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            signed_flags = ctypes.c_int32(flags).value
            zoom_factor = 1.15 if signed_flags > 0 else (1.0 / 1.15)
            
            new_s = self.s * zoom_factor
            if 0.05 <= new_s <= 100.0:
                self.s = new_s
                self.tx = x - self.s * ix
                self.ty = y - self.s * iy
                
        # 3. Clique do Botão Esquerdo
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.hovered_idx is not None:
                self.selected_idx = self.hovered_idx
                self.is_dragging = True
            else:
                self.selected_idx = None
                
        # 4. Arraste e Pan
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.selected_idx is not None:
                img_h, img_w = self.img.shape[:2]
                ix_clamped = max(0.0, min(float(img_w - 1), ix))
                iy_clamped = max(0.0, min(float(img_h - 1), iy))
                self.vertices[self.selected_idx] = [ix_clamped, iy_clamped]
                self.was_adjusted = True
            elif self.is_panning:
                self.tx = self.pan_start_tx + (x - self.pan_start_x)
                self.ty = self.pan_start_ty + (y - self.pan_start_y)
                
        # 5. Soltar Botão Esquerdo
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            
        # 6. Clique do Botão Direito (Pan)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            self.pan_start_tx = self.tx
            self.pan_start_ty = self.ty
            
        # 7. Soltar Botão Direito
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            
        # 8. Duplo-clique Esquerdo (Adicionar novo ponto)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.vertices.append([ix, iy])
            self.selected_idx = len(self.vertices) - 1
            self.was_adjusted = True

    def run(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.window_width, self.window_height)
        cv2.setMouseCallback(
            self.window_title,
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param)
        )
        
        while True:
            M = np.float32([[self.s, 0, self.tx], [0, self.s, self.ty]])
            view = cv2.warpAffine(
                self.img, M, (self.window_width, self.window_height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30)
            )
            
            # Desenhar pontos originais como referência vermelha
            for pt in self.original_points:
                sx, sy = self.to_screen(pt)
                cv2.circle(view, (sx, sy), 3, (100, 100, 240), 1, cv2.LINE_AA)
                
            # Desenhar pontos atuais
            for i, pt in enumerate(self.vertices):
                sx, sy = self.to_screen(pt)
                
                if i == self.selected_idx:
                    color = (50, 100, 255)   # Laranja (Selecionado)
                    radius = self.vertex_radius + 3
                elif i == self.hovered_idx:
                    color = (50, 255, 255)   # Amarelo (Hover)
                    radius = self.vertex_radius + 1
                else:
                    color = (50, 220, 50)    # Verde (Normal)
                    radius = self.vertex_radius
                    
                cv2.circle(view, (sx, sy), radius, color, -1, cv2.LINE_AA)
                cv2.circle(view, (sx, sy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                
            # Desenhar painel de ajuda
            overlay_help = view.copy()
            cv2.rectangle(overlay_help, (10, 10), (self.window_width - 10, 80), (15, 15, 15), -1)
            cv2.addWeighted(overlay_help, 0.75, view, 0.25, 0, view)
            
            instructions = [
                "Ajuste da Grade: Arraste os pontos verdes para as intersecoes da grade a laser no MDF",
                "Mover Ponto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Camera: Click direito + arrastar",
                "Confirmar: [Enter] / [Espaco]               |  Adicionar: Duplo-click  |  Remover: Selecionar + [Delete]/[D]"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(view, text, (20, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
                
            cv2.imshow(self.window_title, view)
            
            key = cv2.waitKey(15)
            if key == -1:
                continue
                
            val = key & 0xFF
            
            if val in [13, 32]:
                self.confirmed = True
                break
            elif val in [27, ord('q'), ord('Q')]:
                self.confirmed = False
                break
            elif val in [ord('r'), ord('R')]:
                self.vertices = [list(pt) for pt in self.initial_vertices]
                self.selected_idx = None
                self.s = self.default_s
                self.tx = self.default_tx
                self.ty = self.default_ty
                self.was_adjusted = False
            elif val in [ord('d'), ord('D')] or key in [3014656, 65535, 46, 2424832, 127]:
                if self.selected_idx is not None:
                    self.vertices.pop(self.selected_idx)
                    self.selected_idx = None
                    self.was_adjusted = True
                    
        cv2.destroyWindow(self.window_title)
        
        if self.confirmed:
            return np.array(self.vertices, dtype=np.float32), True
        else:
            return None, False


def adjust_grid_points(image, auto_points, window_title="Ajuste da Grade de Calibracao", cache_key=None):
    """
    Interface publica para acionar o ajuste manual dos pontos da grade de calibracao.
    Possui cache para evitar multiplas confirmacoes do mesmo background.
    """
    if cache_key is not None and cache_key in _ADJUSTED_GRID_CACHE:
        logger.info(f"Usando grade de calibracao obtida do cache para: {cache_key}")
        return _ADJUSTED_GRID_CACHE[cache_key], True
        
    try:
        editor = InteractiveGridEditor(image, auto_points, window_title)
        points, adjusted = editor.run()
        if cache_key is not None:
            _ADJUSTED_GRID_CACHE[cache_key] = points
        return points, adjusted
    except Exception as e:
        logger.error(f"Erro no ajuste manual dos pontos da grade: {e}. Mantendo originais.")
        return auto_points, False


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRAÇÃO MANUAL VIA BORDAS DO MDF
# ══════════════════════════════════════════════════════════════════════════════

class Interactive4CornerSelector:
    def __init__(self, image, window_title="Calibracao Manual", help_instructions=None, labels=None, initial_corners=None):
        self.img = image.copy()
        self.window_title = window_title
        
        h, w = self.img.shape[:2]
        
        # Se cantos iniciais forem fornecidos, usá-los, senão inicializar com retângulo padrão
        if initial_corners is not None and len(initial_corners) == 4:
            self.vertices = [list(pt) for pt in initial_corners]
        else:
            cw, ch = w // 2, h // 2
            rw, rh = w // 4, h // 4
            self.vertices = [
                [cw - rw, ch - rh],  # Canto Superior Esquerdo
                [cw + rw, ch - rh],  # Canto Superior Direito
                [cw + rw, ch + rh],  # Canto Inferior Direito
                [cw - rw, ch + rh]   # Canto Inferior Esquerdo
            ]
            
        self.initial_vertices = [list(pt) for pt in self.vertices]
        
        # Dimensões da janela
        self.window_width = config.INTERACTIVE_WINDOW_WIDTH
        aspect = h / w
        self.window_height = int(self.window_width * aspect)
        if self.window_height > 900:
            self.window_height = 900
            self.window_width = int(self.window_height / aspect)
            
        # Parâmetros de Zoom e Pan
        scale_w = self.window_width / w
        scale_h = self.window_height / h
        self.s = min(scale_w, scale_h) * 0.95
        self.tx = (self.window_width - w * self.s) / 2
        self.ty = (self.window_height - h * self.s) / 2
        
        self.default_s = self.s
        self.default_tx = self.tx
        self.default_ty = self.ty
        
        self.vertex_radius = config.INTERACTIVE_VERTEX_RADIUS
        self.snap_distance = config.INTERACTIVE_SNAP_DISTANCE
        
        self.hovered_idx = None
        self.selected_idx = None
        self.is_dragging = False
        self.is_panning = False
        
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_tx = 0.0
        self.pan_start_ty = 0.0
        
        self.confirmed = False
        
        # Mensagens de ajuda e rótulos personalizáveis
        if help_instructions is None:
            self.help_instructions = [
                "Calibracao Manual: Arraste os 4 cantos",
                "Mover Canto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Camera: Click direito + arrastar",
                "Confirmar: [Enter] / [Espaco]               |  Resetar Cantos: [R]  |  Cancelar: [Esc] / [Q]"
            ]
        else:
            self.help_instructions = help_instructions
            
        if labels is None:
            self.labels = ["TL", "TR", "BR", "BL"]
        else:
            self.labels = labels

    def to_screen(self, pt):
        x_s = self.s * pt[0] + self.tx
        y_s = self.s * pt[1] + self.ty
        return int(round(x_s)), int(round(y_s))

    def to_image(self, screen_pt):
        x_i = (screen_pt[0] - self.tx) / self.s
        y_i = (screen_pt[1] - self.ty) / self.s
        return x_i, y_i

    def mouse_callback(self, event, x, y, flags, param):
        ix, iy = self.to_image((x, y))
        
        # Hover
        self.hovered_idx = None
        min_dist = float('inf')
        for i, pt in enumerate(self.vertices):
            sx, sy = self.to_screen(pt)
            dist = np.hypot(x - sx, y - sy)
            if dist < self.snap_distance and dist < min_dist:
                min_dist = dist
                self.hovered_idx = i
                
        # Zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            signed_flags = ctypes.c_int32(flags).value
            zoom_factor = 1.15 if signed_flags > 0 else (1.0 / 1.15)
            
            new_s = self.s * zoom_factor
            if 0.05 <= new_s <= 100.0:
                self.s = new_s
                self.tx = x - self.s * ix
                self.ty = y - self.s * iy
                
        # Click Esquerdo
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.hovered_idx is not None:
                self.selected_idx = self.hovered_idx
                self.is_dragging = True
                
        # Drag / Pan
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.selected_idx is not None:
                img_h, img_w = self.img.shape[:2]
                ix_clamped = max(0.0, min(float(img_w - 1), ix))
                iy_clamped = max(0.0, min(float(img_h - 1), iy))
                self.vertices[self.selected_idx] = [ix_clamped, iy_clamped]
            elif self.is_panning:
                self.tx = self.pan_start_tx + (x - self.pan_start_x)
                self.ty = self.pan_start_ty + (y - self.pan_start_y)
                
        # Release Esquerdo
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            
        # Click Direito
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            self.pan_start_tx = self.tx
            self.pan_start_ty = self.ty
            
        # Release Direito
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False

    def run(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.window_width, self.window_height)
        cv2.setMouseCallback(
            self.window_title,
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param)
        )
        
        while True:
            M = np.float32([[self.s, 0, self.tx], [0, self.s, self.ty]])
            view = cv2.warpAffine(
                self.img, M, (self.window_width, self.window_height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30)
            )
            
            # Desenhar as arestas conectando os 4 cantos
            screen_pts = [self.to_screen(pt) for pt in self.vertices]
            for i in range(4):
                pt1 = screen_pts[i]
                pt2 = screen_pts[(i + 1) % 4]
                cv2.line(view, pt1, pt2, (80, 220, 80), 2, cv2.LINE_AA)
                
            # Desenhar os vértices com labels correspondentes
            for i, pt in enumerate(screen_pts):
                if i == self.selected_idx:
                    color = (50, 100, 255)
                    radius = self.vertex_radius + 4
                elif i == self.hovered_idx:
                    color = (50, 255, 255)
                    radius = self.vertex_radius + 2
                else:
                    color = (240, 150, 50)
                    radius = self.vertex_radius
                    
                cv2.circle(view, pt, radius, color, -1, cv2.LINE_AA)
                cv2.circle(view, pt, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Exibir texto identificando o canto
                if i < len(self.labels):
                    cv2.putText(view, self.labels[i], (pt[0] + 12, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
                
            # Desenhar painel de ajuda
            overlay_help = view.copy()
            cv2.rectangle(overlay_help, (10, 10), (self.window_width - 10, 80), (15, 15, 15), -1)
            cv2.addWeighted(overlay_help, 0.75, view, 0.25, 0, view)
            
            for i, text in enumerate(self.help_instructions):
                cv2.putText(view, text, (20, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
                
            cv2.imshow(self.window_title, view)
            
            key = cv2.waitKey(15)
            if key == -1:
                continue
                
            val = key & 0xFF
            
            if val in [13, 32]:
                self.confirmed = True
                break
            elif val in [27, ord('q'), ord('Q')]:
                self.confirmed = False
                break
            elif val in [ord('r'), ord('R')]:
                self.vertices = [list(pt) for pt in self.initial_vertices]
                self.selected_idx = None
                self.s = self.default_s
                self.tx = self.default_tx
                self.ty = self.default_ty
                
        cv2.destroyWindow(self.window_title)
        
        if self.confirmed:
            return np.array(self.vertices, dtype=np.float32)
        else:
            return None


class InteractiveMdfCalibrator(Interactive4CornerSelector):
    def __init__(self, image, window_title="Calibracao Manual via Borda do MDF"):
        help_instructions = [
            "Calibracao Manual: Arraste os 4 cantos para as 4 quinas externas da placa de MDF (300x230 mm)",
            "Mover Canto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Camera: Click direito + arrastar",
            "Confirmar: [Enter] / [Espaco]               |  Resetar Cantos: [R]  |  Cancelar: [Esc] / [Q]"
        ]
        labels = ["TL (Superior-Esquerdo)", "TR (Superior-Direito)", "BR (Inferior-Direito)", "BL (Inferior-Esquerdo)"]
        super().__init__(image, window_title=window_title, help_instructions=help_instructions, labels=labels)


class InteractiveCheckerboardGridEditor:
    def __init__(self, image, points, pattern_size=(14, 10), window_title="Validacao da Malha do Checkerboard"):
        self.img = image.copy()
        self.window_title = window_title
        self.pattern_size = pattern_size
        self.cols, self.rows = pattern_size
        
        # Garantir pontos com formato adequado
        self.original_points = np.array(points, dtype=np.float32).reshape(-1, 2)
        self.vertices = [list(pt) for pt in self.original_points]
        self.initial_vertices = [list(pt) for pt in self.original_points]
        
        # Dimensões da janela ajustadas
        img_h, img_w = self.img.shape[:2]
        self.window_width = config.INTERACTIVE_WINDOW_WIDTH
        aspect = img_h / img_w
        self.window_height = int(self.window_width * aspect)
        if self.window_height > 900:
            self.window_height = 900
            self.window_width = int(self.window_height / aspect)
            
        # Parâmetros de Zoom e Pan
        scale_w = self.window_width / img_w
        scale_h = self.window_height / img_h
        self.s = min(scale_w, scale_h) * 0.95
        self.tx = (self.window_width - img_w * self.s) / 2
        self.ty = (self.window_height - img_h * self.s) / 2
        
        self.default_s = self.s
        self.default_tx = self.tx
        self.default_ty = self.ty
        
        # Configurações visuais (menor para grade densa)
        self.vertex_radius = 4
        self.snap_distance = 12
        
        # Estados
        self.hovered_idx = None
        self.selected_idx = None
        self.is_dragging = False
        self.is_panning = False
        
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_tx = 0.0
        self.pan_start_ty = 0.0
        
        self.confirmed = False
        self.was_adjusted = False
        
    def to_screen(self, pt):
        x_s = self.s * pt[0] + self.tx
        y_s = self.s * pt[1] + self.ty
        return int(round(x_s)), int(round(y_s))

    def to_image(self, screen_pt):
        x_i = (screen_pt[0] - self.tx) / self.s
        y_i = (screen_pt[1] - self.ty) / self.s
        return x_i, y_i

    def mouse_callback(self, event, x, y, flags, param):
        ix, iy = self.to_image((x, y))
        
        # 1. Hover
        self.hovered_idx = None
        min_dist = float('inf')
        for i, pt in enumerate(self.vertices):
            sx, sy = self.to_screen(pt)
            dist = np.hypot(x - sx, y - sy)
            if dist < self.snap_distance and dist < min_dist:
                min_dist = dist
                self.hovered_idx = i
                
        # 2. Zoom via Wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            signed_flags = ctypes.c_int32(flags).value
            zoom_factor = 1.15 if signed_flags > 0 else (1.0 / 1.15)
            
            new_s = self.s * zoom_factor
            if 0.05 <= new_s <= 100.0:
                self.s = new_s
                self.tx = x - self.s * ix
                self.ty = y - self.s * iy
                
        # 3. Clique do Botão Esquerdo
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.hovered_idx is not None:
                self.selected_idx = self.hovered_idx
                self.is_dragging = True
            else:
                self.selected_idx = None
                
        # 4. Arraste e Pan
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.selected_idx is not None:
                img_h, img_w = self.img.shape[:2]
                ix_clamped = max(0.0, min(float(img_w - 1), ix))
                iy_clamped = max(0.0, min(float(img_h - 1), iy))
                self.vertices[self.selected_idx] = [ix_clamped, iy_clamped]
                self.was_adjusted = True
            elif self.is_panning:
                self.tx = self.pan_start_tx + (x - self.pan_start_x)
                self.ty = self.pan_start_ty + (y - self.pan_start_y)
                
        # 5. Soltar Botão Esquerdo
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            
        # 6. Clique do Botão Direito (Pan)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            self.pan_start_tx = self.tx
            self.pan_start_ty = self.ty
            
        # 7. Soltar Botão Direito
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False

    def run(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.window_width, self.window_height)
        cv2.setMouseCallback(
            self.window_title,
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param)
        )
        
        while True:
            M = np.float32([[self.s, 0, self.tx], [0, self.s, self.ty]])
            view = cv2.warpAffine(
                self.img, M, (self.window_width, self.window_height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30)
            )
            
            # 1. Desenhar a malha de conexões entre os pontos (linhas sutis)
            screen_pts = [self.to_screen(pt) for pt in self.vertices]
            for r in range(self.rows):
                for c in range(self.cols):
                    idx = r * self.cols + c
                    if idx >= len(screen_pts):
                        continue
                    pt_curr = screen_pts[idx]
                    
                    # Conectar à direita
                    if c < self.cols - 1:
                        idx_r = r * self.cols + (c + 1)
                        if idx_r < len(screen_pts):
                            cv2.line(view, pt_curr, screen_pts[idx_r], (200, 200, 100), 1, cv2.LINE_AA)
                            
                    # Conectar abaixo
                    if r < self.rows - 1:
                        idx_d = (r + 1) * self.cols + c
                        if idx_d < len(screen_pts):
                            cv2.line(view, pt_curr, screen_pts[idx_d], (200, 200, 100), 1, cv2.LINE_AA)
                            
            # 2. Desenhar pontos originais como referência vermelha
            for pt in self.original_points:
                sx, sy = self.to_screen(pt)
                cv2.circle(view, (sx, sy), 2, (100, 100, 240), 1, cv2.LINE_AA)
                
            # 3. Desenhar pontos atuais
            for i, pt in enumerate(self.vertices):
                sx, sy = screen_pts[i]
                
                if i == self.selected_idx:
                    color = (50, 100, 255)   # Laranja (Selecionado)
                    radius = self.vertex_radius + 3
                elif i == self.hovered_idx:
                    color = (50, 255, 255)   # Amarelo (Hover)
                    radius = self.vertex_radius + 1
                else:
                    color = (50, 220, 50)    # Verde (Normal)
                    radius = self.vertex_radius
                    
                cv2.circle(view, (sx, sy), radius, color, -1, cv2.LINE_AA)
                cv2.circle(view, (sx, sy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                
            # 4. Desenhar painel de ajuda
            overlay_help = view.copy()
            cv2.rectangle(overlay_help, (10, 10), (self.window_width - 10, 80), (15, 15, 15), -1)
            cv2.addWeighted(overlay_help, 0.75, view, 0.25, 0, view)
            
            instructions = [
                f"Validacao da Malha ({self.cols}x{self.rows}): Arraste pontos verdes para ajustar as intersecoes do xadrez",
                "Mover Ponto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Camera: Click direito + arrastar",
                "Confirmar Malha: [Enter] / [Espaco]         |  Resetar: [R]  |  Cancelar (Usar auto/anterior): [Esc] / [Q]"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(view, text, (20, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
                
            cv2.imshow(self.window_title, view)
            
            key = cv2.waitKey(15)
            if key == -1:
                continue
                
            val = key & 0xFF
            
            if val in [13, 32]:
                self.confirmed = True
                break
            elif val in [27, ord('q'), ord('Q')]:
                self.confirmed = False
                break
            elif val in [ord('r'), ord('R')]:
                self.vertices = [list(pt) for pt in self.initial_vertices]
                self.selected_idx = None
                self.s = self.default_s
                self.tx = self.default_tx
                self.ty = self.default_ty
                self.was_adjusted = False
                
        cv2.destroyWindow(self.window_title)
        
        if self.confirmed:
            return np.array(self.vertices, dtype=np.float32), True
        else:
            return self.original_points, False


def select_checkerboard_corners_manually(image, window_title="Calibracao Manual do Checkerboard", cache_key=None, initial_corners=None):
    """
    Interface publica para acionar o seletor manual de 4 cantos do checkerboard.
    Retorna os 4 cantos selecionados pelo usuario ou None se cancelado.
    """
    if cache_key is not None and cache_key in _MANUAL_CORNERS_CACHE:
        logger.info(f"Usando cantos do checkerboard obtidos do cache para: {cache_key}")
        return _MANUAL_CORNERS_CACHE[cache_key]
        
    try:
        help_instructions = [
            "Calibracao Manual: Arraste os 4 cantos para as 4 quinas internas mais externas do xadrez",
            "Mover Canto: Click esquerdo + arrastar  |  Zoom: Scroll Mouse  |  Camera: Click direito + arrastar",
            "Confirmar: [Enter] / [Espaco]               |  Resetar Cantos: [R]  |  Cancelar: [Esc] / [Q]"
        ]
        labels = [
            "TL (Superior-Esquerdo Interno)",
            "TR (Superior-Direito Interno)",
            "BR (Inferior-Direito Interno)",
            "BL (Inferior-Esquerdo Interno)"
        ]
        editor = Interactive4CornerSelector(
            image,
            window_title=window_title,
            help_instructions=help_instructions,
            labels=labels,
            initial_corners=initial_corners
        )
        corners = editor.run()
        if cache_key is not None and corners is not None:
            _MANUAL_CORNERS_CACHE[cache_key] = corners
        return corners
    except Exception as e:
        logger.error(f"Erro na selecao manual dos cantos do checkerboard: {e}")
        return None


def fine_tune_checkerboard_grid(image, points, pattern_size=(14, 10), window_title="Validacao da Malha do Checkerboard", cache_key=None):
    """
    Interface publica para acionar o editor interativo de 140 pontos para o ajuste fino do checkerboard.
    Possui cache para evitar multiplas confirmacoes do mesmo background.
    """
    try:
        editor = InteractiveCheckerboardGridEditor(image, points, pattern_size, window_title)
        points, adjusted = editor.run()
        return points, adjusted
    except Exception as e:
        logger.error(f"Erro no ajuste fino dos pontos da grade do checkerboard: {e}. Mantendo originais.")
        return points, False


def calibrate_mdf_manually(image, window_title="Calibracao Manual via Borda do MDF", cache_key=None):
    """
    Interface publica para acionar o calibrador manual de 4 cantos do MDF.
    Possui cache para evitar multiplas confirmacoes do mesmo background.
    """
    if cache_key is not None and cache_key in _MANUAL_CORNERS_CACHE:
        logger.info(f"Usando cantos de calibracao manual do cache para: {cache_key}")
        return _MANUAL_CORNERS_CACHE[cache_key]
        
    try:
        editor = InteractiveMdfCalibrator(image, window_title)
        corners = editor.run()
        if cache_key is not None and corners is not None:
            _MANUAL_CORNERS_CACHE[cache_key] = corners
        return corners
    except Exception as e:
        logger.error(f"Erro na calibracao manual dos cantos do MDF: {e}")
        return None

