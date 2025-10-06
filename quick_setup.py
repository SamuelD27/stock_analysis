

import os
import sys
import subprocess

def main():
    venv_dir = "venv"
    requirements_file = "requirements.txt"

    print("=== Quick Setup Script ===")

    # 1. Create virtual environment
    if not os.path.isdir(venv_dir):
        print(f"Creating virtual environment in '{venv_dir}'...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("Virtual environment created.")
    else:
        print(f"Virtual environment '{venv_dir}' already exists.")

    # 2. Detect OS and provide activation command
    if os.name == "nt":
        activate_cmd = r".\venv\Scripts\activate"
        python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_bin = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        activate_cmd = "source ./venv/bin/activate"
        python_bin = os.path.join(venv_dir, "bin", "python")
        pip_bin = os.path.join(venv_dir, "bin", "pip")

    print(f"To activate the virtual environment, run:\n  {activate_cmd}")

    # 3. Install dependencies from requirements.txt automatically after activation
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from '{requirements_file}'...")
        try:
            subprocess.check_call([pip_bin, "install", "-r", requirements_file])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("Error installing dependencies. Please check your requirements.txt.")
    else:
        print(f"'{requirements_file}' not found. Skipping dependency installation.")

    print("Setup complete.")

if __name__ == "__main__":
    main()