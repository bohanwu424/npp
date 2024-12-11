#! /bin/bash
python exact_demo.py --skew=10 --repeats=10 --n-end=200 --n-len=3 --h-prior-scale=1 --h-prior-offset=0.2 --nsamples=100
# python exact_demo.py --skew=10 --repeats=40 --n-end=200 --n-len=10 --h-prior-scale=1 --h-prior-offset=0.2 --nsamples=1000 --spike-prob=0.2
# -W error::RuntimeWarning