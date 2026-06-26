import traceback
import sys
import cv2
import numpy as np
sys.path.append('d:\\GitHub\\OpenCV')
from ceramic_analysis import analysis

metrics = [{
    'contour': np.array([[[0,0]], [[10,0]], [[10,10]], [[0,10]]], dtype=np.int32),
    'sample_id': 'IMG_1_P1',
    'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 10, 'bbox_h': 10
}]
img = np.zeros((100,100,3), dtype=np.uint8)

try:
    print("Running annotate_image_multiple...")
    analysis.annotate_image_multiple(img, metrics, 'test_ann.png')
    print("Success")
except Exception as e:
    traceback.print_exc()
