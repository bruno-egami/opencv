file_path = r'd:\GitHub\OpenCV\ceramic_analysis\analysis.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'y_pos = int(y + metrics.get("bbox_h", 0) / 2)',
    'y_pos = int(y + metrics.get("bbox_h", 0) + 25 * font_scale)'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
