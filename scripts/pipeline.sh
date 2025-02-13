#!/bin/bash

# Set variables
DADA_KEY="dada"  # Replace with your actual DADA key
BUFFER_SIZE=$((6160*77784))  # Adjust as needed
NUM_BLOCKS=8
CAPTURE_DURATION=10  # Duration in seconds, adjust as needed
OUTPUT_DIR="./"  # Replace with your desired output path
DIR=/home/user/connor/code/casm_xengine/src/
DUMPFIL_PATH=$DIR"/dumpfil"  # Replace with the actual path to your dumpfil executable
CAPTURE_PATH=$DIR"/dsaX_capture"  # Replace with the actual path to your dsaX_capture executable
CONTROL_IP=127.0.0.1
DATA_IP=192.168.1.3
DATA_PORT=20000
CORRCONF=$DIR/correlator_header_dsaX.txt

# Function to clean up and exit
cleanup() {
    echo "Cleaning up..."
    pkill -f "dada_db -k $DADA_KEY"
    pkill -f "$DUMPFIL_PATH"
    pkill -f "$CAPTURE_PATH"
    exit
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT

# Create DADA buffer
echo "Creating DADA buffer..."
dada_db -k $DADA_KEY -b $BUFFER_SIZE -n $NUM_BLOCKS -l -p

# Start dumpfil in the background
echo "Starting dumpfil..."
$DUMPFIL_PATH -f "${OUTPUT_DIR}/output.fil" -i $DADA_KEY -n 4&

DUMPFIL_PID=$!

# Give dumpfil a moment to start up
sleep 5

# Start packet capture
echo "Starting packet capture..."
$CAPTURE_PATH -c 0 -f $CORRCONF -i $CONTROL_IP -j $DATA_IP -p $DATA_PORT -o $DADA_KEY &
CAPTURE_PID=$!

# Wait for the specified duration
echo "Capturing data for $CAPTURE_DURATION seconds..."
sleep $CAPTURE_DURATION

# Stop processes
echo "Stopping capture and dumpfil..."
kill $CAPTURE_PID
kill $DUMPFIL_PID

# Wait for processes to finish
wait $CAPTURE_PID
wait $DUMPFIL_PID

echo "Pipeline completed successfully."
