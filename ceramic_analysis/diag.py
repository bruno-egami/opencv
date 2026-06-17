# -*- coding: utf-8 -*-
"""Diagnostic script for segmentation issues."""
import cv2
import numpy as np

bg = cv2.imread('data/sessions/session_Teste-prisma-56x38x9/converted/background/top/IMG_20260615_221228.jpg')
img = cv2.imread('data/sessions/session_Teste-prisma-56x38x9/converted/dry/top/IMG_20260615_221242.jpg')

print(f'Background shape: {bg.shape}')
print(f'Specimen shape:   {img.shape}')
bg_orient = "portrait" if bg.shape[0] > bg.shape[1] else "landscape"
img_orient = "portrait" if img.shape[0] > img.shape[1] else "landscape"
print(f'Background orientation: {bg_orient}')
print(f'Specimen orientation:   {img_orient}')

h, w = img.shape[:2]

# Check center region of specimen for HSV values
cy, cx = h // 2, w // 2
roi_size = 100
roi = img[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
print(f'\nCenter ROI (specimen) HSV stats:')
print(f'  H: mean={np.mean(hsv_roi[:,:,0]):.1f}, min={np.min(hsv_roi[:,:,0])}, max={np.max(hsv_roi[:,:,0])}')
print(f'  S: mean={np.mean(hsv_roi[:,:,1]):.1f}, min={np.min(hsv_roi[:,:,1])}, max={np.max(hsv_roi[:,:,1])}')
print(f'  V: mean={np.mean(hsv_roi[:,:,2]):.1f}, min={np.min(hsv_roi[:,:,2])}, max={np.max(hsv_roi[:,:,2])}')

# MDF background region (upper-right corner of the MDF, away from specimen)
# Since the MDF base is visible in the specimen image, pick a corner within MDF
# Looking at the image, MDF covers roughly the central 60% of the image
mdf_roi_y = int(h * 0.2)
mdf_roi_x = int(w * 0.7)
corner_roi = img[mdf_roi_y:mdf_roi_y + 100, mdf_roi_x:mdf_roi_x + 100]
hsv_corner = cv2.cvtColor(corner_roi, cv2.COLOR_BGR2HSV)
print(f'\nMDF ROI (away from specimen) HSV stats:')
print(f'  H: mean={np.mean(hsv_corner[:,:,0]):.1f}, min={np.min(hsv_corner[:,:,0])}, max={np.max(hsv_corner[:,:,0])}')
print(f'  S: mean={np.mean(hsv_corner[:,:,1]):.1f}, min={np.min(hsv_corner[:,:,1])}, max={np.max(hsv_corner[:,:,1])}')
print(f'  V: mean={np.mean(hsv_corner[:,:,2]):.1f}, min={np.min(hsv_corner[:,:,2])}, max={np.max(hsv_corner[:,:,2])}')

# Full image saturation analysis
hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
sat = hsv_full[:, :, 1]
print(f'\nFull image Saturation stats:')
print(f'  mean={np.mean(sat):.1f}, max={np.max(sat)}, std={np.std(sat):.1f}')

# Strategy 1: High saturation mask (yellow piece has higher saturation)
for sat_thresh in [60, 70, 80, 90, 100]:
    sat_mask = (sat > sat_thresh).astype(np.uint8) * 255
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    print(f'\nSaturation > {sat_thresh}: {len(large_contours)} large contours (>5000px)')
    for i, c in enumerate(large_contours[:5]):
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        cx2 = x + cw // 2
        cy2 = y + ch // 2
        print(f'  Contour {i}: bbox=({x},{y},{cw},{ch}), area={area:.0f}px2, center=({cx2},{cy2})')

# Strategy 2: LAB color space analysis
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_ch, a_ch, b_ch = cv2.split(lab)
print(f'\nLAB stats - Center (specimen):')
lab_roi = lab[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]
print(f'  L: mean={np.mean(lab_roi[:,:,0]):.1f}')
print(f'  A: mean={np.mean(lab_roi[:,:,1]):.1f}')
print(f'  B: mean={np.mean(lab_roi[:,:,2]):.1f}')

print(f'\nLAB stats - MDF background:')
lab_corner = lab[mdf_roi_y:mdf_roi_y + 100, mdf_roi_x:mdf_roi_x + 100]
print(f'  L: mean={np.mean(lab_corner[:,:,0]):.1f}')
print(f'  A: mean={np.mean(lab_corner[:,:,1]):.1f}')
print(f'  B: mean={np.mean(lab_corner[:,:,2]):.1f}')

# B channel difference (yellow vs brown)
b_diff = float(np.mean(lab_roi[:,:,2])) - float(np.mean(lab_corner[:,:,2]))
print(f'\n  B channel difference (specimen - MDF): {b_diff:.1f}')

# Try B channel threshold
b_mean = np.mean(b_ch)
b_std = np.std(b_ch)
print(f'\nB channel: mean={b_mean:.1f}, std={b_std:.1f}')
for b_thresh in [b_mean + b_std, b_mean + 1.5 * b_std, b_mean + 2 * b_std]:
    b_mask = (b_ch > b_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    print(f'\nB > {b_thresh:.1f}: {len(large_contours)} large contours')
    for i, c in enumerate(large_contours[:5]):
        x, y, cw, ch2 = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        print(f'  Contour {i}: bbox=({x},{y},{cw},{ch2}), area={area:.0f}px2')

# Strategy 3: Combined approach - Saturation + B channel
combined = cv2.bitwise_and(
    (sat > 70).astype(np.uint8) * 255,
    (b_ch > b_mean + b_std).astype(np.uint8) * 255
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
print(f'\nCombined (Sat>70 AND B>{b_mean + b_std:.1f}): {len(large_contours)} large contours')
for i, c in enumerate(large_contours[:5]):
    x, y, cw, ch2 = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    rw, rh = rect[1]
    print(f'  Contour {i}: bbox=({x},{y},{cw},{ch2}), area={area:.0f}px2, minRect={rw:.0f}x{rh:.0f}px')

# Save diagnostic images
cv2.imwrite('output/diag_sat_mask.png', (sat > 80).astype(np.uint8) * 255)
cv2.imwrite('output/diag_b_mask.png', (b_ch > b_mean + b_std).astype(np.uint8) * 255)
cv2.imwrite('output/diag_combined.png', combined)
print('\nDiagnostic images saved to output/')
