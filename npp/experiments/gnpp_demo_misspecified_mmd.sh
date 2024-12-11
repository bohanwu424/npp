#!/bin/sh
python gB_demo.py \
--parameter-samples=10 --likelihood-samples=match --repeats=10 --n-end=500 --n-len=3 \
--skew=10 --spike-prob=0.1 \
--rate=0.6  --diverge=mmd --kernel='imq'