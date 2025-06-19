import os
import subprocess
import time
import yaml

import snap_setup

def run_pipeline(start_snap=True, 
                 use_dada_dbdisk=False, 
                 use_correlator=True):
    """ This is a preliminary function for running the 
    test pipeline. It can program the SNAP and start sending packets,
    create a DADA buffer, capture packets and place them in that 
    DADA buffer, and then run the real-time correlator. 
    """
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
    CAPTURE_PATH = os.path.join(DIR, "casm_capture")
    CONTROL_IP = config['CONTROL_IP']
    DATA_IP = config['DATA_IP']
    SNAP_IP = config['SNAP_IP']
    SNAP_FPG = config['SNAP_FPG']
    NIC_MAC = config['NIC_MAC']
    NCHAN_PER_PACKET = config['NCHAN_PER_PACKET']
    DATA_PORT = config['DATA_PORT']
    CORRCONF = os.path.join(DIR, "correlator_header_dsaX.txt")

    # Function to clean up and exit
    def cleanup():
        print("Cleaning up...")
        subprocess.run(["dada_db", "-k", {DADA_KEY}, "-d"])
        subprocess.run(["pkill", "-f", CORRELATOR_PATH])
        subprocess.run(["pkill", "-f", CAPTURE_PATH])
        exit()

    # Set up trap to call cleanup function on script exit
    import atexit
    atexit.register(cleanup)

    if start_snap:
        snap_setup.setup_snap(SNAP_IP,
                   fn=SNAP_FPG,
                   DATA_IP=DATA_IP,
                   NIC_MAC=NIC_MAC,
                   set_zeros=False,
                   NCHAN_PER_PACKET=NCHAN_PER_PACKET)

    # Create DADA buffer
    print("Creating DADA buffer...")
    subprocess.run(["dada_db", "-k", DADA_KEY, "-b", str(BUFFER_SIZE), "-n", str(NUM_BLOCKS), "-l", "-p"])

    # Run the specified process
    if use_dada_dbdisk:
        print("Running dada_dbdisk...")
        subprocess.run(["dada_dbdisk", "-k", DADA_KEY, "-D", OUTPUT_DIR])
    
    if use_correlator:
        print("Running casm_correlator...")
        subprocess.run(["sudo", CORRELATOR_PATH, "-k", DADA_KEY])

    # Give dumpfil a moment to start up
    time.sleep(5)

    # Start packet capture
    print("Starting packet capture...")
    capture_process = subprocess.Popen([CAPTURE_PATH, "-c", "0", "-f", CORRCONF, "-i", CONTROL_IP, "-j", DATA_IP, "-p", str(DATA_PORT), "-o", DADA_KEY, "-d"])
#    os.system('sudo ../src/casm_capture -c 1 -f ../src/correlator_header_dsaX.txt -j 192.168.0.1 -i 127.0.0.1 -p 10000 -q 1000 -o dada')

    # Wait for the specified duration
    print(f"Capturing data for {CAPTURE_DURATION} seconds...")
    time.sleep(CAPTURE_DURATION)

    # Cleanup
    cleanup()

if __name__ == "__main__":
    run_pipeline()
