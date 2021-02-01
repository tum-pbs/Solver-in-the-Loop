# ----------------------------------------------------------------------------
#
# Phiflow Burgers 2D Test Makefile
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Burgers 2D examples
#
# ----------------------------------------------------------------------------

SHELL:=/bin/bash

################################################################################
# Reference

# training set
burgers-fdt-hires-set: burgers.py
	for i in `seq 0 9`; do \
		python $^ -o $@ -r 128 -l 32 --dt 0.1 --skipsteps 30 -t 200 --gpu "-1" --seed $$i --thumb; \
	done

burgers-nof-fdt-hires-set: burgers.py
	for i in `seq 0 9`; do \
		python $^ -o $@ -r 128 -l 32 --dt 0.1 --skipsteps 30 -t 200 --gpu "-1" --seed $$i --noforce --thumb; \
	done

# test set
burgers-fdt-hires-testset: burgers.py
	for i in `seq 0 4`; do \
		python $^ -o $@ -r 128 -l 32 --dt 0.1 --skipsteps 30 -t 200 --gpu "-1" --seed 10$$i --thumb; \
	done

burgers-nof-fdt-hires-testset: burgers.py
	for i in `seq 0 4`; do \
		python $^ -o $@ -r 128 -l 32 --dt 0.1 --skipsteps 30 -t 200 --gpu "-1" --seed 10$$i --noforce --thumb; \
	done

################################################################################
# Source (will not be used for training)

# Basic (start from the downsampled hires state)
burgers-fdt-lores-set: burgers.py
	for i in `seq 0 9`; do \
		python $^ -o $@ -r 32 -l 32 --dt 0.1 --skipsteps 0 -t 200 -d 4 --seed $$i \
			--initvH burgers-fdt-hires-set/sim_`printf "%06d" $$i`/velo_000000.npz \
			--loadfH "burgers-fdt-hires-set/sim_`printf "%06d" $$i`/forc_0*.npz" \
			--gpu "-1" --thumb; \
	done

burgers-fdt-lores-testset: burgers.py
	for i in `seq 0 4`; do \
		python $^ -o $@ -r 32 -l 32 --dt 0.1 --skipsteps 0 -t 200 -d 4 --seed 10$$i \
			--initvH burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/velo_000000.npz \
			--loadfH "burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/forc_0*.npz" \
			--gpu "-1" --thumb; \
	done

################################################################################
# Training models

# NON
burgers-fdt-non: burgers_train.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs 100 --lr 0.0001 -l 32 --dt 0.1 -t 200 -s 4 -m 1 -n 10 -b 5 --seed 0 --gpu '0' --cuda \
		--train burgers-fdt-hires-set

burgers-nof-fdt-non: burgers_train.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs 100 --lr 0.0001 -l 32 --dt 0.1 -t 200 -s 4 -m 1 -n 10 -b 5 --seed 0 --gpu '0' --cuda \
		--train burgers-fdt-hires-set --noforce

# SOL-04
burgers-fdt-sol04: burgers_train.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs 100 --lr 0.0001 -l 32 --dt 0.1 -t 200 -s 4 -m 4 -n 10 -b 5 --seed 0 --gpu '0' --cuda \
		--train burgers-fdt-hires-set

burgers-nof-fdt-sol04: burgers_train.py
	python $^ --tf $@/tf --log $@/tf/run.log --epochs 100 --lr 0.0001 -l 32 --dt 0.1 -t 200 -s 4 -m 4 -n 10 -b 5 --seed 0 --gpu '0' --cuda \
		--train burgers-fdt-hires-set --noforce

################################################################################
# Run tests

# NON
burgers-fdt-non/run_test: burgers_apply.py
	for i in `seq 0 4`; do \
		python $^ -o $@ --stats `dirname $@`/tf/dataStats.pickle \
			--model `dirname $@`/tf/model.h5 --gpu "-1" \
			--initvH burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/velo_000000.npz \
			--loadfH "burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/forc_0*.npz" \
			-s 4 -r 32 -l 32 --dt 0.1 -t 200 --thumb; \
	done

# SOL-04
burgers-fdt-sol04/run_test: burgers_apply.py
	for i in `seq 0 4`; do \
		python $^ -o $@ --stats `dirname $@`/tf/dataStats.pickle \
			--model `dirname $@`/tf/model.h5 --gpu "-1" \
			--initvH burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/velo_000000.npz \
			--loadfH "burgers-fdt-hires-testset/sim_`printf "%06d" $$i`/forc_0*.npz" \
			-s 4 -r 32 -l 32 --dt 0.1 -t 200 --thumb; \
	done
