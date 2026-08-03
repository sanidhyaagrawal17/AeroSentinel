import os

for f in os.listdir():
    if "\\" in f:
        if f.endswith("\\"):
            os.remove(f)
            continue
        new_f = f.replace("\\", "/")
        dir_name = os.path.dirname(new_f)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        os.rename(f, new_f)
print("Paths fixed!")
