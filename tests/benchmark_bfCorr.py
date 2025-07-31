import os 

dir = '/home/user/software/casm_xengine/src'
in_key = 'daaa'
out_key = 'dddd'

os.system(f"dada_junkdb -k {in_key} -t 3600 header.txt")
os.system(f"dada_db -k {in_key} -b 1073741824 -n 4")
os.system(f"dada_db -k {out_key} -b 16777216 -n 4")
os.system(f"{dir}/casm_bfCorr -b -i {in_key} \
          -o {out_key} -f {dir}/empty.flags \
          -a {dir}/dummy.calib -p {dir}/powers.out")