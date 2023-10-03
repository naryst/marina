#!/usr/bin/env bash

cd ./../
echo "Git revision: "
git rev-parse HEAD

export test_name=w8a
export use_vr_marina=True
mpiexec -n 6 python ./experiment_vr_all.py
