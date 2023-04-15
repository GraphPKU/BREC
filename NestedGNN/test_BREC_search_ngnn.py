import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python test_BREC.py --h=1 --layers=6 --width=16 --OUTPUT_DIM=16 --BATCH_SIZE=32 --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-5 --node_label=no --SEED={seed}'
    print(script_base)
    os.system(script_base)