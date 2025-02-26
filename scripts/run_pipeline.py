import os
import subprocess
import time
import yaml

def run_pipeline(use_dada_dbdisk=False, use_correlator=True):
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set variables from config
    DADA_KEY = config['DADA_KEY']
    BUFFER_SIZE = config['BUFFER_SIZE']
    NUM_BLOCKS = config['NUM_BLOCKS']
    CAPTURE_DURATION = config['CAPTURE_DURATION']
    OUTPUT_DIR = config['OUTPUT_DIR']
    DIR = config['DIR']
    CORRELATOR_PATH = os.path.join(DIR, "casm_correlator")
    CAPTURE_PATH = os.path.join(DIR, "dsaX_capture")
    CONTROL_IP = config['CONTROL_IP']
    DATA_IP = config['DATA_IP']
    DATA_PORT = config['DATA_PORT']
    CORRCONF = os.path.join(DIR, "correlator_header_dsaX.txt")

    # Function to clean up and exit
    def cleanup():
        print("Cleaning up...")
        subprocess.run(["pkill", "-f", f"dada_db -k {DADA_KEY}"])
        subprocess.run(["pkill", "-f", CORRELATOR_PATH])
        subprocess.run(["pkill", "-f", CAPTURE_PATH])
        exit()

    # Set up trap to call cleanup function on script exit
    import atexit
    atexit.register(cleanup)

    # Create DADA buffer
    print("Creating DADA buffer...")
    subprocess.run(["dada_db", "-k", DADA_KEY, "-b", str(BUFFER_SIZE), "-n", str(NUM_BLOCKS), "-l", "-p"])

    # Run the specified process
    if use_dada_dbdisk:
        print("Running dada_dbdisk...")
        subprocess.run(["dada_dbdisk", "-k", DADA_KEY, "-D", OUTPUT_DIR])
    
    if use_correlator:
        print("Running casm_correlator...")
        subprocess.run([echo, CORRELATOR_PATH, "-k", DADA_KEY, "-D", OUTPUT_DIR])

    # Give dumpfil a moment to start up
    time.sleep(5)

    # Start packet capture
    print("Starting packet capture...")
    capture_process = subprocess.Popen([CAPTURE_PATH, "-c", "0", "-f", CORRCONF, "-i", CONTROL_IP, "-j", DATA_IP, "-p", str(DATA_PORT), "-o", DADA_KEY, "-d"])

    # Wait for the specified duration
    print(f"Capturing data for {CAPTURE_DURATION} seconds...")
    time.sleep(CAPTURE_DURATION)

    # Cleanup
    cleanup()

if __name__ == "__main__":
    run_pipeline()
