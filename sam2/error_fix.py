import os
import subprocess
import sys

venv_path = sys.prefix  # root of your virtualenv
matches = []

for root, dirs, files in os.walk(venv_path):
    for f in files:
        if f.endswith(".so") or f.endswith(".dylib"):
            full_path = os.path.join(root, f)
            try:
                output = subprocess.check_output(["otool", "-L", full_path], text=True)
            except subprocess.CalledProcessError:
                continue
            if "libomp" in output:
                matches.append((full_path, output.strip()))

# Print results
if matches:
    print("=== Libraries linked against libomp.dylib ===\n")
    for path, otool_out in matches:
        print(f"File: {path}")
        print(otool_out)
        print("-" * 60)
else:
    print("No .so/.dylib files linked to libomp.dylib were found in this venv.")
