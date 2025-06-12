# R.A.P.T.O.R Setup Script
# File: setup_raptor.py

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print R.A.P.T.O.R setup banner"""
    banner = """
🦅 ═══════════════════════════════════════════════════════════════════════════ 🦅
     
    ██████╗  █████╗ ██████╗ ████████╗ ██████╗ ██████╗ 
    ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
    ██████╔╝███████║██████╔╝   ██║   ██║   ██║██████╔╝
    ██╔══██╗██╔══██║██╔═══╝    ██║   ██║   ██║██╔══██╗
    ██║  ██║██║  ██║██║        ██║   ╚██████╔╝██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝
    
    Real-time Aerial Patrol and Tactical Object Recognition
    Advanced AI-Powered Tactical Detection System
    
🦅 ═══════════════════════════════════════════════════════════════════════════ 🦅
"""
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        print("Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("🖥️ Checking system requirements...")
    
    system = platform.system()
    print(f"Operating System: {system} {platform.release()}")
    
    # Check available memory (basic check)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️ Warning: Less than 4GB RAM available. Performance may be limited.")
        else:
            print("✅ Sufficient memory available")
    except ImportError:
        print("ℹ️ Cannot check memory (psutil not available)")
    
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing R.A.P.T.O.R requirements...")
    
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "geopandas>=0.13.0",
        "shapely>=2.0.0",
        "pyproj>=3.5.0",
        "pytest>=7.0.0",
        "requests>=2.30.0",
        "python-dateutil>=2.8.0"
    ]
    
    # Optional requirements
    optional_requirements = [
        "psutil",  # For system monitoring
        "black",   # Code formatting
        "flake8"   # Code linting
    ]
    
    failed_packages = []
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    # Install optional packages (non-critical)
    print("\n📦 Installing optional packages...")
    for package in optional_requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ Optional package {package} failed to install (non-critical)")
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install these manually or check your internet connection.")
        return False
    
    print("\n✅ All requirements installed successfully!")
    return True

def create_directory_structure():
    """Create R.A.P.T.O.R directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "src",
        "tests", 
        "docs",
        "output",
        "output/detections",
        "output/images",
        "output/videos", 
        "output/maps",
        "output/analysis",
        "output/testing",
        "data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    print("✅ Directory structure created!")
    return True

def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check system requirements
    if not check_system_requirements():
        print("⚠️ System requirements check failed, but continuing...")
    
    # Step 3: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        sys.exit(1)
    
    # Step 4: Create directory structure
    create_directory_structure()
    
    print("🎉 R.A.P.T.O.R setup completed successfully!")
    print("🚀 System ready for tactical operations!")

if __name__ == "__main__":
    main()