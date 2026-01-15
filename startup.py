"""
Startup script - automatically sets up venv and installs dependencies if needed.
Run this file to start the application: python startup.py
"""

import subprocess
import sys
import os

VENV_DIR = "venv"
REQUIREMENTS = ["pygame"]


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def get_venv_pip():
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")


def venv_exists():
    """Check if the virtual environment exists."""
    return os.path.exists(get_venv_python())


def venv_is_healthy():
    """Check if the virtual environment is functional."""
    python_path = get_venv_python()
    if not os.path.exists(python_path):
        return False
    try:
        subprocess.check_call(
            [python_path, "-c", "import sys; sys.exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def remove_venv():
    """Remove the existing virtual environment."""
    import shutil
    if os.path.exists(VENV_DIR):
        print("Removing broken virtual environment...")
        shutil.rmtree(VENV_DIR)
        print("Removed.")


def create_venv():
    """Create a new virtual environment."""
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    print("Virtual environment created.")


def install_requirements():
    """Install required packages in the virtual environment."""
    python_path = get_venv_python()
    print("Installing dependencies...")
    # Use python -m pip to avoid issues with pip upgrading itself on Windows
    subprocess.check_call([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    for package in REQUIREMENTS:
        print(f"Installing {package}...")
        subprocess.check_call([python_path, "-m", "pip", "install", package])
    print("All dependencies installed.")


def check_packages_installed():
    """Check if all required packages are installed."""
    python_path = get_venv_python()
    for package in REQUIREMENTS:
        try:
            subprocess.check_call(
                [python_path, "-c", f"import {package}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return False
    return True


def run_main():
    """Run the main application using the virtual environment's Python."""
    python_path = get_venv_python()
    print("Starting application...\n")
    subprocess.call([python_path, "main.py"])


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check if venv exists but is broken
    if venv_exists() and not venv_is_healthy():
        remove_venv()

    # Create venv if it doesn't exist
    if not venv_exists():
        create_venv()
        install_requirements()
    elif not check_packages_installed():
        print("Some packages are missing.")
        install_requirements()

    run_main()


if __name__ == "__main__":
    main()
