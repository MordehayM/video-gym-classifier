import os
import json

classes = sorted(os.listdir('datafolder'))

class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}

output = {
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class
}

with open("class_mappings.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved to class_mappings.json")