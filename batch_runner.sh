#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT="model_best.pth.tar"

trap "exit" INT

get_checkpoint_path () {
    cd sparse_to_dense
    output_folder="$(python -c 'import utils; print(utils.get_output_directory(utils.parse_command()));' $cmd)"
    cd ..
    checkpoint_path="$output_folder/$CHECKPOINT"
}

while true; do

    # find any interrupted commands
    cmd="$(grep -P -m 1 '^[^#]*.(#\ *started)' commands.txt)"

    if [[ -z "$cmd" ]]; then # cmd is empty
        # find new command to run
        cmd="$(grep -P -m 1 '^[^#]*.$' commands.txt)"
        if [[ -z "$cmd" ]]; then # still no commands
            echo "no more commands to run"
            break # no more commands to run
        fi
        sed -i -e "/${cmd}\ *$/s/$/ # started/" commands.txt # set started
        echo "running: python sparse_to_dense/main.py $cmd"
        python sparse_to_dense/main.py $cmd
    else # resume
        get_checkpoint_path 
        # check if there is a checkpoint available
        if [[ -f "$checkpoint_path" ]]; then
            echo "resuming $checkpoint_path"
            python sparse_to_dense/main.py --resume $checkpoint_path
        else
            echo "no checkpoint, running: python sparse_to_dense/main.py $cmd"
            python sparse_to_dense/main.py $cmd 
        fi
    fi

    retval=$?
    if [ $retval -ne 0 ]; then
        echo "failed $ln"
        sed -i -e "/${cmd}/s/started/failed/" commands.txt 

    else
        echo "------------- finished, doing final calculations -------------"
        get_checkpoint_path 
        python sparse_to_dense/main.py --evaluate_tum $checkpoint_path
        echo "finished $ln"
        sed -i -e "/${cmd}/s/started/finished/" commands.txt 
    fi
done