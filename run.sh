# 
list_name="vpg ddpg trpo ppo td3 sac" 

if [ -n $1 ] && [[ "$list_name" =~ "$1" ]]; then 
    echo "执行程序: $1"
    proc=$1
    xvfb-run -s "-screen 0 640x480x24" python $proc/run.py 
else
    echo "没有该选项, [$list_name]"
fi

