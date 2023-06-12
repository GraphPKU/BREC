import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python test_BREC.py with configs/brec/edge_del/del1_subgraph20_imle_{seed}.yaml'
    print(script_base)
    os.system(script_base)
