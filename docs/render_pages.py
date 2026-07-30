import fitz

doc = fitz.open(r'D:\Downloads\AeroSentinel_InnoVent2026_Cygnus.pdf')
out_dir = r'..'\docs\pdf_pages'

import os
os.makedirs(out_dir, exist_ok=True)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    pix.save(os.path.join(out_dir, f'page_{i+1:02d}.png'))
    print(f'Saved page {i+1}')

print('Done')
