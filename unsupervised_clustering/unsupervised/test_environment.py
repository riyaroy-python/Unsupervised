import sys
import importlib

def get_required_packages():
    required_packages = {}
    with open('requirements.txt', 'r') as f:
        for line in f:
            package_info = line.strip().split('==')
            if len(package_info) == 2:
                package_name, package_version = package_info
                required_packages[package_name] = package_version
    return required_packages

def check_package_versions():
    required_packages = get_required_packages()
    for package_name, required_version in required_packages.items():
        try:
            module = importlib.import_module(package_name)
            installed_version = module.__version__
            if installed_version != required_version:
                print(f"Warning: {package_name} version {installed_version} is installed, but version {required_version} is required.")
        except ImportError:
            print(f"Error: {package_name} is not installed.")

    print("Environment check completed.")

if __name__ == '__main__':
    check_package_versions()
    sys.exit(0)