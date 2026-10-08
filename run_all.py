import subprocess
import sys

notebooks = [
    "notebooks/Chapter3_CNN_Diabetes.ipynb",
    "notebooks/Chapter3_CNN_VietnamHousing.ipynb",
    "notebooks/Chapter4_RNN_Gold.ipynb"
]

for nb in notebooks:
    print(f"Executing {nb}...")
    result = subprocess.run([
        r"C:\Users\Admin\AppData\anaconda3\Scripts\jupyter.exe", 
        "nbconvert", 
        "--execute", 
        "--inplace", 
        nb
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: {nb}")
    else:
        print(f"FAILED: {nb}")
        print(result.stderr)
