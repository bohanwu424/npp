#!/bin/sh
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="gnpp-demo/full_$TIMESTAMP.txt"
EXPERIMENTSFILE="experiments/gnpp_demo_full_args.txt"
JOBS=32
echo $OUTFILE
# --dry-run
parallel --rpl '{} uq()' --jobs $JOBS -a $EXPERIMENTSFILE \
python gB_demo.py --log $OUTFILE \
--parameter-samples=100 --likelihood-samples=match --repeats=100 --seed=1234 \
--n-start=5 --n-end=500 --n-len=10 \
--spike-prob=0.1 --kernel=imq --wass-p=2 --transform \
{}