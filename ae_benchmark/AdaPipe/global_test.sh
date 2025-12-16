#!/bin/bash

cd "$(dirname "$0")"

hostfile=$1

echo "test gpt"
ws=64
sh gpt_experiment.sh $ws 128 4096 $hostfile
sh gpt_experiment.sh $ws 64 8192 $hostfile
sh gpt_experiment.sh $ws 32 16384 $hostfile

echo "test llama2"
ws=32
sh llama2_experiment.sh $ws 128 4096 $hostfile
sh llama2_experiment.sh $ws 64 8192 $hostfile
sh llama2_experiment.sh $ws 32 16384 $hostfile
