import shutil, os

folder = "data/features"
shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder, exist_ok=True)

print("🧹 Cleaned: data/features (fresh start)")
