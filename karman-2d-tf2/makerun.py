# ----------------------------------------------------------------------------
#
# Phiflow Karman 2D Test Makefile
# Copyright 2020-2021 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Karman 2D examples
#
# ----------------------------------------------------------------------------

import pathlib
import subprocess
import os

cwd = os.getcwd().replace(" ", "%20").replace("\\", "/")
save_path = cwd

################################################################################
# Reference

# training set
# karman-fdt-hires-set

filename = "karman.py"
for i in range(0, 6):
    subprocess.run(
        f"python karman-2d-tf2/{filename} -o {save_path} -r 128 -l 100 --re {((10_000 * 2**(i+4)))} --gpu '-1' --seed 0 --thumb"
    )


# test set
# karman-fdt-hires-testset
filename = "karman.py"
for i in range(0, 5):
    exec(
        f"python {filename} -o {save_path} -r 128 -l 100 --re {((10_000 * 2**(i+3) * 3))} --gpu '-1' --seed 0 --thumb"
    )

################################################################################
# Source (will not be used for training)

# karman-fdt-lores-set:
filename = "karman.py"
for i in range(0, 6):
    exec(
        f"python {filename} -o {save_path} -r 32  -l 100 --re {((10_000 * 2**(i+4)))} --gpu '-1' --seed 0 --thumb --skipsteps 0 -t 500 -d 4"
        + f"--initdH karman-fdt-hires-set/sim_{i:06d}/dens_001000.npz"
        + f"--initvH karman-fdt-hires-set/sim_{i:06d}/velo_001000.npz"
    )


# karman-fdt-lores-testset
filename = "karman.py"
for i in range(0, 5):
    exec(
        f"python {filename} -o {save_path} -r 32  -l 100 --re {((10_000 * 2**(i+3) * 3 ))} --gpu '-1' --seed 0 --thumb --skipsteps 0 -t 500 -d 4"
        + f"--initdH karman-fdt-hires-testset/sim_{i:06d}/dens_001000.npz"
        + f"--initvH karman-fdt-hires-testset/sim_{i:06d}/velo_001000.npz"
    )

################################################################################
# Training models

# NON
# karman-fdt-non
filename = "karman_train.py"  # NOTE: don't test with "-n 1" (normalization problem because std(Re)=0!)
exec(
    f"python {filename} --tf {save_path}/tf --log {save_path}/tf/run.log --epochs=100 --lr 0.0001 -l 100 -t 500 -s 4 -m 1 -n 6 -b 3 --seed 0 --gpu '0' --cuda \
		--train karman-fdt-hires-set"
)

# SOL-08
# karman-fdt-sol08:
filename = "karman_train.py"  # NOTE: don't test with "-n 1" (normalization problem because std(Re)=0!)
exec(
    f"python {filename} --tf {save_path}/tf --log {save_path}/tf/run.log --epochs=100 --lr 0.0001 -l 100 -t 500 -s 4 -m 8 -n 6 -b 3 --seed 0 --gpu '0' --cuda \
		--train karman-fdt-hires-set"
)

################################################################################
# Run tests

# NON
# karman-fdt-non/run_test
filename = "karman_apply.py"
for i in range(0, 5):
    exec(
        f"python {filename} -o {save_path} --thumb \
        --stats {savepath}/tf/dataStats.pickle \
        --model {savepath}/tf/model.h5 --gpu '-1' \
        --initdH karman-fdt-hires-testset/sim_{i:06d}`/dens_001000.npz \
        --initvH karman-fdt-hires-testset/sim_{i:06d}`/velo_001000.npz \
        -s 4 -r 32 -l 100 --re {((10_000 * 2**(i+3) * 3 ))} -t 500"
    )

# SOL-08
# karman-fdt-sol08/run_test
filename = "karman_apply.py"
for i in range(0, 5):
    exec(
        f"python {filename} -o {save_path} --thumb \
        --stats {save_path}/tf/dataStats.pickle \
        --model {save_path}/tf/model.h5 --gpu '-1' \
        --initdH karman-fdt-hires-testset/sim_{i:06d}/dens_001000.npz \
        --initvH karman-fdt-hires-testset/sim_{i:06d}/velo_001000.npz \
        -s 4 -r 32 -l 100 --re {(( 10_000 * 2**(i+3) * 3 ))} -t 500"
    )
