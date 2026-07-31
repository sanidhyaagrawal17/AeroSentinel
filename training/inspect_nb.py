import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\AeroSentinel TATA AIG 2026\training\notebooka2ae66b251.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 9 is the markdown header for Cell 5
cell9 = nb['cells'][9]
print("=== Cell 9 markdown ===")
print(''.join(cell9['source']))

# Also check outputs of Cell 10 to see the actual output
cell10 = nb['cells'][10]
if cell10.get('outputs'):
    print("\n=== Cell 10 outputs ===")
    for out in cell10['outputs']:
        if out.get('text'):
            print(''.join(out['text']))
        elif out.get('data', {}).get('text/plain'):
            print(''.join(out['data']['text/plain']))
