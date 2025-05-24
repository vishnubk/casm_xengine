import numpy as np

def read_visibilities(filename, nchan=128, nants=12):
    """
    Reads the binary file containing averaged visibilities.
    
    Each integration period is stored as:
      nchan x (nants x nants x 2 doubles)
    where the last dimension contains [Re, Im] for each matrix element.
    
    Returns:
      A complex numpy array with shape:
          (num_integrations, nchan, nants, nants)
    """
    # Read the entire file as a 1D array of doubles
    data = np.fromfile(filename, dtype=np.float64)
    
    # Calculate the number of doubles per integration period
    doubles_per_integration = nchan * nants * nants * 2
    if data.size % doubles_per_integration != 0:
        raise ValueError("File size is not a multiple of a single integration period's data.")
    
    num_integrations = data.size // doubles_per_integration
    print(f"Found {num_integrations} integration periods in the file.")
    
    # Reshape to [num_integrations, nchan, nants, nants, 2]
    data = data.reshape((num_integrations, nchan, nants, nants, 2))
    
    # Combine the last dimension into a complex number: real + 1j*imag
    visibilities = data[..., 0] + 1j * data[..., 1]
    
    return visibilities

if __name__ == "__main__":
    filename = "visibilities.bin"
    try:
        vis = read_visibilities(filename, nchan=3072)
        print("Visibilities shape:", vis.shape)
        # For example, display the first integration period for channel 0:
        print("Integration period 0, channel 0 visibility matrix:")
        print(vis[0, 0])
    except Exception as e:
        print("Error reading file:", e)
