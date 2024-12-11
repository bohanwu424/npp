#!/bin/sh
python gB_demo.py \
--parameter-samples=10 --repeats=100 --n-end=500 --n-len=3 \
--skew=10 --spike-prob=0.1 \
--rate=0.99 --diverge=ksd --kernel=imq --transform