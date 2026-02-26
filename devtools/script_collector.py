import os
import sys

from src.constants import ROOT_DIR, SRC_DIR


def collect_scripts(source_files, output_file):
    """
    Combines multiple Python scripts into one file.
    - Moves all imports to the top.
    - Removes local imports (e.g. 'from src.agent import ...').
    - Preserves the order of the source files content.
    """
    
    # Sets to track unique imports
    future_imports = set()
    imports = []
    seen_imports = set()
    
    # List to hold the code blocks from each file
    code_blocks = []
    
    # Prefixes for local imports that should be removed in the combined file
    # (Since the code is now in the same file, we don't need to import from src.*)
    local_import_prefixes = ["from src.", "import src."]

    for file_path in source_files:
        filename = os.path.basename(file_path)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        file_code = []
        file_code.append("\n"*2)
        file_code.append(f"\n# {'='*50}\n")
        file_code.append(f"# SECTION: {filename}\n")
        file_code.append(f"# {'='*50}\n")
        
        for line in lines:
            stripped = line.strip()
            
            # 1. Handle Imports
            if line.startswith("import ") or line.startswith("from "):
                # Check if it is a local import to remove
                if any(prefix in line for prefix in local_import_prefixes):
                    file_code.append(f"# [REMOVED LOCAL IMPORT] {stripped}\n")
                    continue
                
                # Check for __future__ imports
                if "from __future__" in line:
                    future_imports.add(line)
                else:
                    # Deduplicate regular imports
                    if stripped not in seen_imports:
                        imports.append(line)
                        seen_imports.add(stripped)
            
            # 2. Handle Regular Code
            else:
                file_code.append(line)
        
        code_blocks.append("".join(file_code))

    # Write the combined file
    with open(output_file, "w", encoding="utf-8") as f:
        # Write future imports first
        if future_imports:
            f.writelines(sorted(list(future_imports)))
            f.write("\n")
            
        # Write other imports
        f.write("# --- Combined Imports ---\n")
        f.writelines(imports)
        f.write("\n")
        
        # Write code blocks
        for block in code_blocks:
            f.write(block)
            
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    # Define the project root and source directory
    root_dir = ROOT_DIR
    src_dir = SRC_DIR
    
    # Define the order of files to combine. 
    # Dependencies should come before usage.
    files = [
        os.path.join(SRC_DIR, "constants.py"),
        os.path.join(SRC_DIR, "experiments.py"),
        os.path.join(SRC_DIR, "template.py"),
        os.path.join(SRC_DIR, "utils.py"),
        os.path.join(SRC_DIR, "buffer.py"),
        os.path.join(SRC_DIR, "model.py"),
        os.path.join(SRC_DIR, "agent.py"),
        os.path.join(SRC_DIR, "experiment_runner.py"),
        os.path.join(ROOT_DIR, "main.py"),
    ]
    
    output_filename = os.path.join(ROOT_DIR, "devtools", "ALL_SCRIPTS.py")
    
    collect_scripts(files, output_filename)