# to run the streamlit app
import subprocess
import sys

def main():
    subprocess.run([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "app/streamlit_app.py"
    ])

if __name__ == "__main__":
    main()
