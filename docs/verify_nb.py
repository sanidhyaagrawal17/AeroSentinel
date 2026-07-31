import json

nb = json.load(open(r'training\AeroSentinel_Training.ipynb', 'r', encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
c3 = ''.join(code_cells[2]['source'])

# FIX 1 checks
checks = {
    'mosaic=0,': True, 'mixup=0,': True, 'copy_paste=0,': True,
    'degrees=0,': True, 'flipud=0,': True, 'freeze=5,': True,
    # Unchanged params
    'fliplr=0.5,': True, 'scale=0.5,': True,
    'hsv_h=0.015,': True, 'hsv_s=0.7,': True, 'hsv_v=0.4,': True,
    'epochs=300,': True, 'device=[0,1],': True, 'batch=16,': True,
    # FIX 2 checks
    'hasattr(results': True,
    'save_dir = PROJECT_ROOT': True,
}

print("=== FIX 1: Augmentation parameter changes ===")
all_ok = True
for k in checks:
    found = k in c3
    status = "OK" if found else "FAIL"
    print(f"  [{status}] {k}")
    if not found:
        all_ok = False

print("\n=== FIX 2: DDP save_dir fix ===")
has_hasattr = 'hasattr(results' in c3
has_fallback = 'save_dir = PROJECT_ROOT' in c3
print(f"  [{'OK' if has_hasattr else 'FAIL'}] hasattr(results, 'save_dir') check present")
print(f"  [{'OK' if has_fallback else 'FAIL'}] Fallback to PROJECT_ROOT / 'runs' path present")

print(f"\n=== Notebook validity ===")
print(f"  Valid JSON: True (parsed successfully)")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Code cells: {len(code_cells)}")

if all_ok and has_hasattr and has_fallback:
    print("\n ALL CHECKS PASSED")
else:
    print("\n SOME CHECKS FAILED")
