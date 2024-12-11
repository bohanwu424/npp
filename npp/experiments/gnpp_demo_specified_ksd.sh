#!/bin/sh
python gB_demo.py \
--parameter-samples=10 --repeats=100 --n-end=500 --n-len=5 \
--skew=0 --spike-prob=0.1 \
--rate=0.99 --diverge=ksd --kernel=imq