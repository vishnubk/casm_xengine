import sys
from subprocess import Popen, PIPE
import re 
from casperfpga import CasperFpga, TapcpTransport
from casm_f import snap_fengine

def setup_snap(IP,
               fn='/home/user/connor/code/casm_snap_f/firmware/snap_f_12i_4kc/outputs/snap_f_12i_4kc_2025-05-08_1446.fpg',
               DATA_IP='192.168.0.1'
               NIC_MAC=0x80615f0c7116,
               NCHAN_PER_PACKET=512):
    """ Here
    """
    f = CasperFpga(IP, transport=TapcpTransport)
    print('Starting initial upload and programming')
    f.upload_to_ram_and_program(fn)

    # Obtaining SNAP board mac
    pid = Popen(["arp","-n",IP],stdout=PIPE)
    s = pid.communicate()[0]
    mac = re.search(r"(([a-f\d]{1,2}\:){5}[a-f\d]{1,2})",str(s)).groups()[0]
    MAC=int(mac.replace(":",""),16)

    # It is critical that the destination mac is correct!
    macs = {IP:MAC, DATA_IP:NIC_MAC}
    dests = [{'ip' : DATA_IP, 
              'port':10000, 
              'start_chan': 0, 
              'nchan' : 3072}]

    # Connecting to snap with casm_f. Use the microblaze rather than the 
    # raspberry Pi connection.
    snap = snap_fengine.SnapFengine(IP, use_microblaze=True)
    snap.program(fn)
        
    print('Now configuring SNAP, starting to send packets.')
    snap.configure(source_ip=IP, source_port=20000, 
                   program=False,  
                   dests=dests, 
                   macs=macs, 
                   nchan_packet=NCHAN_PER_PACKET, 
                   enable_tx=True)
    

def main():
    if len(sys.argv) < 2:
        print("Usage: python snap_setup.py <IP>")
        sys.exit(1)
    ip_0 = sys.argv[1]
    setup_snap(ip_0, set_zeros=False)

if __name__ == '__main__':
    main()
