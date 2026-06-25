import sys

def main():
    file_path = r'd:\GitHub\OpenCV\ceramic_analysis\analysis.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Replace the comment block with the actual drawing code
    target_comment = '''        # As cotas detalhadas (linhas e medidas sobrepostas) foram removidas para evitar
        # poluição visual nas peças, conforme solicitado. As medidas gerais são 
        # apresentadas no texto lateral e no relatório.'''
        
    replacement_code = '''        # Desenhar cotas/linhas de dimensão diretamente sobre a peça no perímetro
        circularity = metrics.get('circularity', 0.0)
        
        if circularity <= 0.80 and 'min_rect_center_x' in metrics and 'min_rect_w' in metrics and 'min_rect_h' in metrics:
            center_pt = (metrics['min_rect_center_x'], metrics['min_rect_center_y'])
            w_px = metrics['min_rect_w']
            h_px = metrics['min_rect_h']
            angle = metrics['min_rect_angle']
            
            box = cv2.boxPoints((center_pt, (w_px, h_px), angle))
            
            px_h = scale.get('px_per_mm_h', 1.0) if scale else 1.0
            px_v = scale.get('px_per_mm_v', 1.0) if scale else 1.0
            
            rect_w_mm = metrics.get('min_rect_w_mm', 0)
            rect_h_mm = metrics.get('min_rect_h_mm', 0)
            major_mm = max(rect_w_mm, rect_h_mm)
            minor_mm = min(rect_w_mm, rect_h_mm)

            drawn_length = False
            drawn_width = False

            for i in range(4):
                p1 = box[i]
                p2 = box[(i + 1) % 4]
                
                dx_mm = (p2[0] - p1[0]) / px_h
                dy_mm = (p2[1] - p1[1]) / px_v
                len_mm = (dx_mm**2 + dy_mm**2)**0.5
                
                if len_mm < 1.0:
                    continue
                    
                if abs(len_mm - major_mm) <= abs(len_mm - minor_mm):
                    if not drawn_length:
                        _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, offset_multiplier=2.0)
                        drawn_length = True
                else:
                    if not drawn_width:
                        _draw_segment_cota(annotated, p1, p2, center_pt, len_mm, font_scale, COLOR_BBOX, COLOR_TEXT, COLOR_TEXT_BG, thickness, offset_multiplier=2.0)
                        drawn_width = True
        elif 'bbox_x' in metrics and 'bbox_y' in metrics and 'bbox_w' in metrics and 'bbox_h' in metrics:
            x, y = metrics['bbox_x'], metrics['bbox_y']
            bw, bh = metrics['bbox_w'], metrics['bbox_h']
            
            offset = int(40 * font_scale)
            tick_size = int(6 * font_scale)
            
            if 'bbox_w_mm' in metrics:
                if y - offset - 10 < 0:
                    dy = bh + offset
                else:
                    dy = -offset
                
                cota_y = y + dy
                cv2.line(annotated, (x, y), (x, cota_y + (5 if dy < 0 else -5)), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x + bw, y), (x + bw, cota_y + (5 if dy < 0 else -5)), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x, cota_y), (x + bw, cota_y), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x - tick_size, cota_y + tick_size), (x + tick_size, cota_y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                cv2.line(annotated, (x + bw - tick_size, cota_y + tick_size), (x + bw + tick_size, cota_y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                text_w = f"{metrics['bbox_w_mm']:.2f} mm"
                _draw_text_with_bg(
                    annotated, text_w, (x + bw // 2, cota_y),
                    font_scale * 0.7, COLOR_TEXT, COLOR_TEXT_BG, max(1, thickness - 1),
                    center=True
                )
                
            if 'bbox_h_mm' in metrics:
                if x - offset - 10 < 0:
                    dx = bw + offset
                else:
                    dx = -offset
                    
                cota_x = x + dx
                cv2.line(annotated, (x, y), (cota_x + (5 if dx < 0 else -5), y), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (x, y + bh), (cota_x + (5 if dx < 0 else -5), y + bh), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (cota_x, y), (cota_x, y + bh), COLOR_BBOX, 1, cv2.LINE_AA)
                cv2.line(annotated, (cota_x - tick_size, y + tick_size), (cota_x + tick_size, y - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                cv2.line(annotated, (cota_x - tick_size, y + bh + tick_size), (cota_x + tick_size, y + bh - tick_size), COLOR_BBOX, 2, cv2.LINE_AA)
                text_h = f"{metrics['bbox_h_mm']:.2f} mm"
                _draw_text_with_bg(
                    annotated, text_h, (cota_x, y + bh // 2),
                    font_scale * 0.7, COLOR_TEXT, COLOR_TEXT_BG, max(1, thickness - 1),
                    center=True
                )'''

    content = content.replace(target_comment, replacement_code)
    
    # 2. Modify annotate_image signature and logic
    target_sig = '''def annotate_image(
    img, metrics, output_path,
    scale=None,
    draw_ellipse=False,
    draw_dimensions=True,
    draw_contour=True,
    draw_bbox=True,
    dim_background=True
):'''
    new_sig = '''def annotate_image(
    img, metrics, output_path=None,
    scale=None,
    draw_ellipse=False,
    draw_dimensions=True,
    draw_contour=True,
    draw_bbox=True,
    dim_background=True,
    img_copy=True
):'''
    content = content.replace(target_sig, new_sig)
    
    target_copy = '''    # Fazer cópia
    annotated = img.copy()

    if dim_background:
        annotated = cv2.convertScaleAbs(annotated, alpha=0.6, beta=0)'''
    new_copy = '''    # Fazer cópia
    annotated = img.copy() if img_copy else img

    if dim_background:
        annotated = cv2.convertScaleAbs(annotated, alpha=0.6, beta=0)'''
    content = content.replace(target_copy, new_copy)
    
    target_save = '''    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.debug(f"  Imagem anotada salva: {output_path}")

    return annotated'''
    new_save = '''    # Salvar
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        logger.debug(f"  Imagem anotada salva: {output_path}")

    return annotated'''
    content = content.replace(target_save, new_save)
    
    # 3. Add annotate_image_multiple to the end
    append_func = '''

def annotate_image_multiple(
    img, metrics_list, output_path,
    scale=None,
    draw_dimensions=True,
    draw_contour=True,
    draw_bbox=True
):
    """
    Anota uma única imagem com os contornos e dimensões de múltiplas peças.
    Escurece o fundo apenas uma vez.
    """
    annotated = img.copy()
    
    # Escurecer o fundo uma única vez
    annotated = cv2.convertScaleAbs(annotated, alpha=0.6, beta=0)

    for metrics in metrics_list:
        draw_ellipse = metrics.get("circularity", 0) > 0.80
        
        annotate_image(
            annotated, metrics, output_path=None,
            scale=scale,
            draw_ellipse=draw_ellipse,
            draw_dimensions=draw_dimensions,
            draw_contour=draw_contour,
            draw_bbox=draw_bbox,
            dim_background=False,
            img_copy=False
        )
        
    if output_path is not None:
        from pathlib import Path
        import cv2
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        logger.debug(f"  Imagem anotada salva (múltipla): {output_path}")

    return annotated
'''
    content = content + append_func
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    main()
