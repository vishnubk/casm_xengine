import sys

import struct
from scapy.all import PcapReader, UDP

def decode_samples(payload, n=10):
    """
    Decode the first n bytes of payload into signed 4-bit real/imag samples.
    Returns a list of complex numbers.
    """
    samples = []
    for i, b in enumerate(payload[:n]):
        # high nibble = real, low nibble = imag
        real = (b >> 4) & 0xF
        imag = b & 0xF
        # Sign‐extend 4-bit -> 8-bit
        if real & 0x8:
            real -= 0x10
        if imag & 0x8:
            imag -= 0x10
        samples.append(complex(real, imag))
    return samples

def read_casm_pcap(pcap_file):
    """
    Stream a .pcap from the CASM F-engine and yield:
      (capture_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload_bytes)
    """
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if UDP in pkt:
                # Capture timestamp (wall-clock) and raw UDP payload
                capture_ts = pkt.time
                raw = bytes(pkt[UDP].payload)
                
                # CASM F-engine header is 16 bytes: 
                #   u64 timestamp; u16 chan0; u16 board_id; u16 n_chans; u16 n_antpols;
                header = raw[:16]
                pkt_ts, chan0, board_id, n_chans, n_antpols = struct.unpack('!QHHHH', header)
                
                # The rest is interleaved 4-bit signed real/imag data:
                payload = raw[16:]
                
                yield capture_ts, pkt_ts, chan0, board_id, n_chans, n_antpols, payload

if __name__ == "__main__":
    fn = sys.argv[1]
    for cap_ts, pkt_ts, chan0, bid, nch, nant, data in read_casm_pcap(fn):
        #print(f"Captured @ {cap_ts:.6f}s | pkt_ts={pkt_ts} | chan0={chan0} | "
        #      f"board_id={bid} | n_chans={nch} | n_antpols={nant} | payload={len(data)} bytes")
        dataarr = decode_samples(data, n=8)
        print(dataarr[0])
