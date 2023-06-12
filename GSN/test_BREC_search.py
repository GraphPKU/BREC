import os

SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = (
        f" python test_BREC.py --mode BREC_test --dataset BREC --dataset_name brec_v3_s4 --root_folder ./datasets "
        f"--id_type all_simple_graphs --induced True --model_name GSN_sparse --msg_kind general "
        f"--num_layers 4 --d_out 64 --wandb False --seed {seed} "
    )

    # GSN-3v
    # script = script_base + " --k 3 --id_scope global"
    # GSN-4v
    # script = script_base + " --k 4 --id_scope global"
    # GSN-3e
    # script = script_base + " --k 3 --id_scope local"
    # GSN-4e
    script = script_base + " --k 4 --id_scope local"
    print(script)
    os.system(script)
