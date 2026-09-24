import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Path to the single JSON file
json_path = os.path.join(project_root, "dataset", "Train", "CAM_Instance_Train.json")

print(f"Reading: {json_path}")
print(f"File exists: {os.path.exists(json_path)}")
print(f"File size: {os.path.getsize(json_path) / (1024*1024):.2f} MB")
print()

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Top-level type: {type(data).__name__}")

if isinstance(data, list):
    print(f"Number of entries: {len(data)}")
    print("\n--- First entry ---")
    print(json.dumps(data[0], indent=2)[:1500])
    print("\n--- Second entry ---")
    print(json.dumps(data[1], indent=2)[:1500])
    
elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())[:10]}")
    first_key = list(data.keys())[0]
    print(f"\n--- First key: {first_key} ---")
    print(json.dumps(data[first_key], indent=2)[:1500])

# Also check the txt 0file
txt_path = os.path.join(project_root, "dataset", "Train", "CAM-NonCAM_Instance_Train.txt")
print(f"\n\n--- TXT file preview ---")
with open(txt_path, 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 4:
            break