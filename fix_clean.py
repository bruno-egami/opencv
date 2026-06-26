import sys

def main():
    file_path = r'd:\GitHub\OpenCV\ceramic_analysis\analysis.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find start and end of annotate_image
    start_str = "def annotate_image("
    end_str = "def _draw_text_with_bg("
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    if start_idx == -1 or end_idx == -1:
        print("Nao achou")
        return
        
    new_func = '''def annotate_image(
    image: np.ndarray,
    metrics: dict,
    output_path: str = None,
    scale: dict = None,
    draw_contour: bool = True,
    draw_bbox: bool = False,
    draw_ellipse: bool = False,
    draw_dimensions: bool = False,
    dim_background: bool = True,
    img_copy: bool = True
) -> np.ndarray:
    import cv2
    import numpy as np
    from pathlib import Path
    
    annotated = image.copy() if img_copy else image
    h, w = annotated.shape[:2]

    if dim_background and "contour" in metrics:
        contour = metrics["contour"]
        mask_inside = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_inside, [contour], -1, 255, -1)
        mask_outside = cv2.bitwise_not(mask_inside)
        
        darkened_bg = cv2.addWeighted(annotated, 0.5, np.zeros_like(annotated), 0.5, 0)
        
        img_inside = cv2.bitwise_and(annotated, annotated, mask=mask_inside)
        img_outside = cv2.bitwise_and(darkened_bg, darkened_bg, mask=mask_outside)
        annotated = cv2.add(img_inside, img_outside)

    COLOR_CONTOUR = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_BG = (0, 0, 0)

    font_scale = max(0.5, min(w, h) / 1500.0)
    thickness = max(1, int(font_scale * 2))

    if draw_contour and "contour" in metrics:
        cv2.drawContours(annotated, [metrics["contour"]], -1, COLOR_CONTOUR, 2)
        
    # Identificacao da peca
    if "sample_id" in metrics:
        text_id = str(metrics["sample_id"]).split("_")[-1] # Apenas "P1", "P2", etc
        
        # Obter bounding box upright para saber onde colocar o texto
        if "bbox_x" in metrics and "bbox_y" in metrics:
            x = metrics["bbox_x"]
            y = metrics["bbox_y"]
            bw = metrics["bbox_w"]
            
            # Tentar colocar acima da peca (y - offset)
            # Se sair da tela, colocar no centro
            y_pos = int(y - 25 * font_scale)
            if y_pos < 10:
                y_pos = int(y + metrics.get("bbox_h", 0) / 2)
                
            x_pos = int(x + bw / 2)
            
            _draw_text_with_bg(
                annotated, text_id, (x_pos, y_pos),
                font_scale * 1.5, COLOR_TEXT, COLOR_TEXT_BG, thickness, center=True
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    return annotated

'''
    
    new_content = content[:start_idx] + new_func + content[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    main()
