import json

try:
    with open('AeroSentinel_Training.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    print("Valid JSON: True")
    
    cell5_src = ''.join(nb['cells'][10]['source'])
    print(f"Contains \\\"\\\"\\\" (proper triple quotes): {'\"\"\"' in cell5_src}")
    print(f"Contains \\\\\"\\\\\"\\\\\" (escaped quotes, should be false): {'\\\"\\\"\\\"' in cell5_src}")
    print(f"Sanity checks present: {'assert box_max > 0' in cell5_src}")

except Exception as e:
    print(f"Error: {e}")
