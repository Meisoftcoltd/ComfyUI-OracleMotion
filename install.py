import os
import subprocess
import sys

def install_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        print(f"Installing requirements from {requirements_path}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    else:
        print("requirements.txt not found. Skipping installation.")

if __name__ == "__main__":
    install_requirements()
