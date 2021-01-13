#!/bin/bash

help_info="h help -h --help"
alg_names="vpg ddpg trpo ppo td3 sac" 
run_types="train plot test"

if [ $# != 2 ] || [[ "$help_info" =~ "$1" ]]; then
    echo $#
    printf "\n命令提示:\nbash run.sh [run_type] [alg_name]\n\n"
    printf "\t--run_type: [train plot test]\n"
    printf "\t--alg_name: [vpg ddpg trpo ppo td3 sac] \n\n"
    exit 1;
else
    type=$1 
    proc=$2
    echo "执行程序: python $proc/run.py $1"
    xvfb-run -s "-screen 0 640x480x24" python  $proc/run.py --run_type $1 --alg $2
fi
# bash ./run.sh train ddpg