import json, re

# Load original
with open(r'training\AeroSentinel_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find code cell 3 (training cell) and change epochs=300 to epochs=15
code_idx = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code_idx += 1
        if code_idx == 3:
            new_source = []
            for line in cell['source']:
                if 'epochs=300,' in line:
                    line = line.replace('epochs=300,', 'epochs=15,')
                new_source.append(line)
            cell['source'] = new_source
            break

# Save as test notebook
out_path = r'training\AeroSentinel_Training_Test.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created {out_path}')

# === Verification ===
nb2 = json.load(open(out_path, 'r', encoding='utf-8'))
code_cells = [c for c in nb2['cells'] if c['cell_type'] == 'code']
c3 = ''.join(code_cells[2]['source'])

checks = {
    'epochs=15,': 'epochs changed to 15',
    'mosaic=0,': 'mosaic=0 unchanged',
    'mixup=0,': 'mixup=0 unchanged',
    'copy_paste=0,': 'copy_paste=0 unchanged',
    'degrees=0,': 'degrees=0 unchanged',
    'flipud=0,': 'flipud=0 unchanged',
    'freeze=5,': 'freeze=5 unchanged',
    'device=[0,1],': 'device=[0,1] unchanged',
    'batch=16,': 'batch=16 unchanged',
    'hasattr(results': 'DDP save_dir fix present',
    'save_dir = PROJECT_ROOT': 'DDP fallback present',
    'fliplr=0.5,': 'fliplr=0.5 unchanged',
    'scale=0.5,': 'scale=0.5 unchanged',
    'hsv_h=0.015,': 'hsv_h unchanged',
    'hsv_s=0.7,': 'hsv_s unchanged',
    'hsv_v=0.4,': 'hsv_v unchanged',
    'patience=75,': 'patience=75 unchanged',
}

# Negative check
no_300 = 'epochs=300,' not in c3

print('\n=== Verification ===')
all_ok = True
for k, desc in checks.items():
    found = k in c3
    status = 'PASS' if found else 'FAIL'
    if not found: all_ok = False
    print(f'  [{status}] {desc}')

print(f'  [{"PASS" if no_300 else "FAIL"}] epochs=300 removed')
if not no_300: all_ok = False

print(f'\n  Valid JSON: True')
print(f'  Total cells: {len(nb2["cells"])}')
print(f'\n  {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
