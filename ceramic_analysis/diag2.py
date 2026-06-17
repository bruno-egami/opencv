# -*- coding: utf-8 -*-
"""Refined diagnostic: find the best segmentation for yellow specimen on MDF."""
import cv2
import numpy as np

bg = cv2.imread('data/sessions/session_Teste-prisma-56x38x9/converted/background/top/IMG_20260615_221228.jpg')
img = cv2.imread('data/sessions/session_Teste-prisma-56x38x9/converted/dry/top/IMG_20260615_221242.jpg')

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# ── Strategy: LAB B channel with adaptive threshold ──
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
b_ch = lab[:, :, 2]

# Use B > mean + 2*std as threshold
b_mean = np.mean(b_ch)
b_std = np.std(b_ch)
b_thresh = b_mean + 2 * b_std
print(f"\nB channel: mean={b_mean:.1f}, std={b_std:.1f}, threshold={b_thresh:.1f}")

b_mask = (b_ch > b_thresh).astype(np.uint8) * 255

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel, iterations=2)

contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [c for c in contours if cv2.contourArea(c) > 10000]
print(f"\nB > {b_thresh:.1f}: {len(large_contours)} large contours (>10000px)")

# Find the contour closest to center with the largest area
cx_img, cy_img = w // 2, h // 2
best_contour = None
best_score = 0

for i, c in enumerate(large_contours):
    x, y, cw, ch2 = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    ccx = x + cw // 2
    ccy = y + ch2 // 2
    
    # Distance from image center
    dist = np.sqrt((ccx - cx_img) ** 2 + (ccy - cy_img) ** 2)
    max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)
    center_score = 1 - (dist / max_dist)
    
    # Score = area * center_proximity
    score = area * center_score
    
    print(f"  Contour {i}: bbox=({x},{y},{cw},{ch2}), area={area:.0f}px2, "
          f"center=({ccx},{ccy}), dist_center={dist:.0f}, score={score:.0f}")
    
    if score > best_score:
        best_score = score
        best_contour = c

if best_contour is not None:
    rect = cv2.minAreaRect(best_contour)
    rw, rh = rect[1]
    angle = rect[2]
    print(f"\nBest contour: minAreaRect = {rw:.1f} x {rh:.1f} px, angle={angle:.1f}°")
    
    # We need a scale factor. Let's use grid detection.
    # For now, use the previous calibration: px_per_mm = 11.5
    px_per_mm = 11.5
    print(f"\nUsing approximate scale: {px_per_mm} px/mm")
    print(f"  Estimated dimensions: {rw/px_per_mm:.1f} x {rh/px_per_mm:.1f} mm")
    print(f"  Expected dimensions: 56.0 x 38.0 mm")
    
    # Draw result
    vis = img.copy()
    cv2.drawContours(vis, [best_contour], -1, (0, 255, 0), 3)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis, [box], -1, (0, 0, 255), 3)
    
    # Scale down for saving
    scale_factor = 0.25
    vis_small = cv2.resize(vis, None, fx=scale_factor, fy=scale_factor)
    cv2.imwrite('output/diag_best_contour.png', vis_small)
    
    # Save the refined mask
    refined_mask = np.zeros_like(b_mask)
    cv2.drawContours(refined_mask, [best_contour], -1, 255, -1)
    cv2.imwrite('output/diag_refined_mask.png', refined_mask)
    
    print("\nSaved: output/diag_best_contour.png, output/diag_refined_mask.png")

# ── Also test improved background subtraction with brightness normalization ──
print("\n\n=== Improved Background Subtraction ===")
# Normalize brightness: scale background to match specimen image brightness in overlapping region
# Use border regions (not center) for normalization
border = 200
border_mask = np.zeros((h, w), dtype=np.uint8)
border_mask[:border, :] = 255
border_mask[-border:, :] = 255
border_mask[:, :border] = 255
border_mask[:, -border:] = 255

bg_border_mean = cv2.mean(bg, mask=border_mask)[:3]
img_border_mean = cv2.mean(img, mask=border_mask)[:3]

print(f"Border mean - BG:  {bg_border_mean}")
print(f"Border mean - IMG: {img_border_mean}")

# Scale each channel
bg_normalized = bg.astype(np.float32)
for ch in range(3):
    if bg_border_mean[ch] > 0:
        bg_normalized[:, :, ch] *= img_border_mean[ch] / bg_border_mean[ch]

bg_normalized = np.clip(bg_normalized, 0, 255).astype(np.uint8)

# Background subtraction on normalized images
diff = cv2.absdiff(img, bg_normalized)
gray_diff = np.max(diff, axis=2)

# Apply Gaussian blur
gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)

print(f"Normalized diff: mean={np.mean(gray_diff):.1f}, max={np.max(gray_diff)}, std={np.std(gray_diff):.1f}")

for thresh in [15, 20, 25, 30]:
    _, mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 10000]
    print(f"\n  Threshold {thresh}: {len(large_contours)} large contours")
    for i, c in enumerate(large_contours[:3]):
        x, y, cw, ch2 = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        print(f"    Contour {i}: bbox=({x},{y},{cw},{ch2}), area={area:.0f}, minRect={rw:.0f}x{rh:.0f}px")

cv2.imwrite('output/diag_norm_diff.png', gray_diff)
cv2.imwrite('output/diag_norm_mask_t25.png', 
            cv2.morphologyEx(
                cv2.morphologyEx(
                    cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)[1],
                    cv2.MORPH_CLOSE, kernel, iterations=3),
                cv2.MORPH_OPEN, kernel, iterations=2))
print("\nSaved normalized diff and mask images")
