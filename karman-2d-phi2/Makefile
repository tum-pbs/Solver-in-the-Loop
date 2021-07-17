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

SHELL:=/bin/bash

################################################################################
# Reference

# training set
karman-fdt-hires-set: karman.py
	for i in `seq 0 5`; do \
		python $^ -o $@ -r 128 -l 100 --re `echo $$(( 10000 * 2**($$i+4) ))` --gpu "-1" --seed 0 --thumb; \
	done

# test set
karman-fdt-hires-testset: karman.py
	for i in `seq 0 4`; do \
		python $^ -o $@ -r 128 -l 100 --re `echo $$(( 10000 * 2**($$i+3) * 3 ))` --gpu "-1" --seed 0 --thumb; \
	done

################################################################################
# Source (will not be used for training)

karman-fdt-lores-set: karman.py
	for i in `seq 0 5`; do \
		python $^ -o $@ -r 32  -l 100 --re `echo $$(( 10000 * 2**($$i+4) ))` --gpu "-1" --seed 0 --thumb --skipsteps 0 -t 500 -d 4 \
			--initdH karman-fdt-hires-set/sim_`printf "%06d" $$i`/dens_001000.npz \
			--initvH karman-fdt-hires-set/sim_`printf "%06d" $$i`/velo_001000.npz; \
	done

karman-fdt-lores-testset: karman.py
	for i in `seq 0 4`; do \
		python $^ -o $@ -r 32  -l 100 --re `echo $$(( 10000 * 2**($$i+3) * 3 ))` --gpu "-1" --seed 0 --thumb --skipsteps 0 -t 500 -d 4 \
			--initdH karman-fdt-hires-testset/sim_`printf "%06d" $$i`/dens_001000.npz \
			--initvH karman-fdt-hires-testset/sim_`printf "%06d" $$i`/velo_001000.npz; \
	done

################################################################################
# Training models

# SOL-08
karman-fdt-sol08: karman_train.py # NOTE: don't test with "-n 1" (normalization problem because std(Re)=0!)
	python $^ --tf $@/tf --log $@/tf/run.log --epochs=100 --lr 0.0001 -r 32 -l 100 -t 500 -s 4 -m 8 -n 6 -b 3 --seed 0 --gpu '0' --skip-ds \
		--train karman-fdt-hires-set

################################################################################
# Run tests

# SOL-08
karman-fdt-sol08/run_test: karman_apply.py
	for i in `seq 0 4`; do \
		python $^ -o $@ --thumb \
			--stats `dirname $@`/tf/dataStats.pickle \
			--model `dirname $@`/tf/model.h5 --gpu "-1" \
			--initdH karman-fdt-hires-testset/sim_`printf "%06d" $$i`/dens_001000.npz \
			--initvH karman-fdt-hires-testset/sim_`printf "%06d" $$i`/velo_001000.npz \
			-r 32 -l 100 --re `echo $$(( 10000 * 2**($$i+3) * 3 ))` -t 500; \
	done
