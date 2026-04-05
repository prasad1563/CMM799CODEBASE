import subprocess
import glob
import os
import sys

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Get all matching notebooks in order
    notebooks = sorted(glob.glob("[0-9][0-9]_*.ipynb"))
    
    if not notebooks:
        print(f"No notebooks found in {script_dir}.")
        return

    print(f"Found {len(notebooks)} notebooks to execute in {script_dir}:")
    for nb in notebooks:
        print(f" - {nb}")
    print("\nStarting execution...\n")

    for nb in notebooks:
        print(f"==================================================")
        print(f"Running {nb}...")
        print(f"==================================================")
        
        # use jupyter nbconvert to execute the notebook in place
        result = subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace", 
            "--ExecutePreprocessor.timeout=-1", # No timeout
            nb
        ])
        
        if result.returncode != 0:
            print(f"\n[ERROR] Execution failed for {nb}. Halting.")
            break
        else:
            print(f"[SUCCESS] Finished {nb}\n")

    print("All notebook executions are complete!")

if __name__ == "__main__":
    main()
