import sys
import argparse
import numpy as np
import struct
from collections import defaultdict
import matplotlib.pyplot as plt
from scapy.all import PcapReader, UDP

def decode_samples(payload, n=6144):
    """
    Decode the first n bytes of payload into signed 4-bit real/imag samples.
    Returns a NumPy array of complex values.
    """
    samples = []
    for i, b in enumerate(payload[:n]):
        real = (b >> 4) & 0xF
        imag = b & 0xF
        if real & 0x8:
            real -= 0x10
        if imag & 0x8:
            imag -= 0x10
        samples.append(complex(real, imag))
    return np.array(samples, dtype=np.complex64)

def read_casm_pcap(pcap_file):
    """
    Stream a .pcap from the CASM F-engine and yield:
      (capture_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload_bytes)
    """
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if UDP in pkt:
                capture_ts = pkt.time
                raw = bytes(pkt[UDP].payload)
                header = raw[:16]
                pkt_ts, chan0, board_id, n_chans, n_antpols = struct.unpack('!QHHHH', header)
                payload = raw[16:]
                yield capture_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload

def main(pcap_file, max_packets=1000, chan0=0, total_chans=3072, adc_sample_rate=250e6):
    """
    Accumulate and plot average spectra for all 12 ADCs.
    """
    freq0 = 375e6
    chan_width_hz = adc_sample_rate / 4096
    freq_axis_mhz = (freq0 + (chan0 + np.arange(total_chans)) * chan_width_hz) / 1e6

    spectra_sum = defaultdict(lambda: np.zeros(total_chans, dtype=np.float32))
    counts = defaultdict(int)

    pkt_count = 0
    for cap_ts, pkt_ts, pkt_chan0, board_id, n_chans, n_antpols, payload in read_casm_pcap(pcap_file):
        # Skip packets outside desired frequency range
        if pkt_chan0 < chan0 or pkt_chan0 + n_chans > chan0 + total_chans:
            continue

        offset = pkt_chan0 - chan0
        samples = decode_samples(payload)
        if len(samples) != n_chans * n_antpols:
            continue

        spectrum = np.abs(samples).reshape((n_chans, n_antpols))
        for adc in range(n_antpols):
            spectra_sum[adc][offset:offset + n_chans] += spectrum[:, adc]
            counts[adc] += 1

        pkt_count += 1
        if pkt_count >= max_packets:
            break

    # Plotting
    fig, axs = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    axs = axs.flatten()

    for adc in range(12):
        if counts[adc] == 0:
            continue
        avg_spectrum = spectra_sum[adc] / counts[adc]
        axs[adc].plot(freq_axis_mhz, avg_spectrum)
        axs[adc].set_title(f"ADC {adc}")
        axs[adc].set_ylabel("Power")
        axs[adc].grid(True)

    axs[-1].set_xlabel("Frequency [MHz]")
    plt.suptitle(f"Average Spectrum (chan0={chan0}, {total_chans} channels, max_pkts={max_packets})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot time-averaged spectra from CASM F-engine PCAP.")
    parser.add_argument("pcap_file", help="Path to PCAP file")
    parser.add_argument("--max-pkts", type=int, default=1000, help="Max number of packets to process")
    parser.add_argument("--chan0", type=int, default=0, help="Starting frequency channel (default: 0)")
    parser.add_argument("--nchans", type=int, default=3072, help="Number of channels to plot (default: 3072)")
    args = parser.parse_args()

    main(args.pcap_file, max_packets=args.max_pkts, chan0=args.chan0, total_chans=args.nchans)
