import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python test_BREC.py --SEED={seed}'
    print(script_base)
    os.system(script_base)
