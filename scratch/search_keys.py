import os

def search_shell_configs():
    files = [
        "/Users/shreyas/.zshrc",
        "/Users/shreyas/.bash_profile",
        "/Users/shreyas/.bashrc",
        "/Users/shreyas/.profile",
        "/Users/shreyas/.zprofile",
        "/Users/shreyas/.config/zsh/.zshrc",
    ]
    
    print("Scanning shell profile/config files for PRISMTRACE...")
    for filepath in files:
        if os.path.exists(filepath):
            print(f"Checking: {filepath}")
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    content = f.read()
                    if "PRISMTRACE" in content:
                        print(f"  -> Contains PRISMTRACE: {filepath}")
                        for line in content.splitlines():
                            if "PRISMTRACE" in line:
                                print(f"    Line: {line}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    search_shell_configs()
