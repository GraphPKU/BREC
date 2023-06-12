import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python test_BREC.py --model=SSWL_P --max_dis=8 --num_layer=8 --dim_embed=64 --BATCH_SIZE=8 --LEARNING_RATE=1e-5 --WEIGHT_DECAY=1e-5 --device=2 --SEED={seed}'
    print(script_base)
    os.system(script_base)
