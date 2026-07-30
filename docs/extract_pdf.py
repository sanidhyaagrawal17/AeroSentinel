import fitz
import json

doc = fitz.open(r'D:\Downloads\AeroSentinel_InnoVent2026_Cygnus.pdf')
result = []
for i, page in enumerate(doc):
    text = page.get_text()
    result.append({
        'page': i + 1,
        'width': page.rect.width,
        'height': page.rect.height,
        'text': text
    })
    
with open(r'..'\docs\pdf_extract.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    
print(f"Extracted {len(result)} pages")
