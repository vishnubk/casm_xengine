import os
import time

junkdb = True
dir = '/home/user/software/casm_xengine/src'
in_key = 'daaa'
out_key = 'dddd'
in_block_size = 1073741824
out_block_size = 16777216

try:
    os.system(f"dada_db -k {in_key} -d")
    os.system(f"dada_db -k {out_key} -d")
except:
    pass

os.system(f"dada_db -k {in_key} -b {in_block_size} -n 4")
os.system(f"dada_db -k {out_key} -b {out_block_size} -n 4")
time.sleep(1)

if junkdb:
    os.system(f"dada_junkdb -k {in_key} -t 3600 header.txt")
else:
    os.system(f"{dir}/fake_writer &")
time.sleep(1)
os.system(f"{dir}/casm_bfCorr -b -i {in_key} \
          -o {out_key} -f {dir}/empty.flags \
          -a {dir}/dummy.calib -p {dir}/powers.out")
