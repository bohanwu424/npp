#!/bin/sh
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="npp-demo/full_$TIMESTAMP.txt"
EXPERIMENTSFILE="experiments/npp_demo_full_args.txt"
JOBS=32
echo $OUTFILE
# --dry-run
parallel --rpl '{} uq()' --jobs $JOBS -a $EXPERIMENTSFILE \
python exact_demo.py --log $OUTFILE \
--repeats=100 --nsamples=1000 --nheldout=1000 --seed=1234 \
--n-start=5 --n-end=500 --n-len=10 \
--spike-prob=0.1 --h-prior-scale=1 --h-prior-offset=0. \
{}

