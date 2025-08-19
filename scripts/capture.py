from scapy.all import sniff, IP, UDP

def handle_packet(packet):
    if packet.haslayer(IP) and packet.haslayer(UDP):
        ip = packet[IP]
        udp = packet[UDP]
        print(f"{ip.src}:{udp.sport} > {ip.dst}:{udp.dport}, length={len(udp)}")

# Replace "eth0" with the interface you want to capture on
sniff(filter="udp", iface="eth0", prn=handle_packet, store=False)
