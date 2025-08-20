import sys
import numpy as np
import struct
from collections import defaultdict
import matplotlib.pyplot as plt
from scapy.all import PcapReader, UDP

def decode_samples(payload, n=6144):
    """Decode interleaved 4-bit complex samples."""
    samples = []
    for i, b in enumerate(payload[:n]):
        real = (b >> 4) & 0xF
        imag = b & 0xF
        real = real - 0x10 if real & 0x8 else real
        imag = imag - 0x10 if imag & 0x8 else imag
        samples.append(complex(real, imag))
    return np.array(samples, dtype=np.complex64)

def read_casm_pcap(pcap_file):
    """Yield packets from a CASM F-engine PCAP."""
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if UDP in pkt:
                capture_ts = pkt.time
                raw = bytes(pkt[UDP].payload)
                header = raw[:16]
                pkt_ts, chan0, board_id, n_chans, n_antpols = struct.unpack('!QHHHH', header)
                payload = raw[16:]
                yield capture_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload

def main(pcap_file):
    spectra_sum = defaultdict(lambda: np.zeros(3072, dtype=np.float32))
    counts = defaultdict(int)

    for cap_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload in read_casm_pcap(pcap_file):
        samples = decode_samples(payload)
        total_chans = n_chans * n_antpols  # usually 3072 = 256 * 12
        if len(samples) != total_chans:
            continue

        # Reshape to shape (n_chans, n_antpols) = (3072, 12)
        spectrum = np.abs(samples).reshape((n_chans, n_antpols))

        # Accumulate per ADC
        for adc in range(n_antpols):
            spectra_sum[adc][chan0:chan0 + n_chans] += spectrum[:, adc]
            counts[adc] += 1

    # Plotting
    fig, axs = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    axs = axs.flatten()

    for adc in range(12):
        if counts[adc] == 0:
            continue
        avg_spectrum = spectra_sum[adc] / counts[adc]
        axs[adc].plot(avg_spectrum)
        axs[adc].set_title(f"ADC {adc}")
        axs[adc].set_ylabel("Power")
        axs[adc].grid(True)

    axs[-1].set_xlabel("Channel Index")
    plt.suptitle("Average Spectrum for Each ADC (12x 3072 channels)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
