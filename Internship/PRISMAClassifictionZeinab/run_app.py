#!/usr/bin/env python3
"""
PRISMA Classification Streamlit App Launcher
This script checks dependencies and launches the Streamlit application.
"""

import sys
import subprocess
import importlib.util
import os

def check_dependency(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 PRISMA Classification Streamlit App Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Required packages
    required_packages = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'numpy': 'numpy',
        'rasterio': 'rasterio',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    print("\n📦 Checking dependencies...")
    for package_name, import_name in required_packages.items():
        if check_dependency(package_name, import_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\n🔧 Installing missing packages...")
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}")
                    print("Please install manually: pip install " + package)
                    sys.exit(1)
        else:
            print("\n❌ Please install missing packages manually:")
            print("pip install " + " ".join(missing_packages))
            sys.exit(1)
    
    # Check if streamlit_app.py exists
    if not os.path.exists('streamlit_app.py'):
        print("\n❌ streamlit_app.py not found!")
        print("Please ensure you're in the correct directory.")
        sys.exit(1)
    
    print("\n✅ All dependencies are satisfied!")
    print("🚀 Launching Streamlit application...")
    print("\n" + "=" * 50)
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
