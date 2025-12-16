#!/bin/bash

ws=$1
gbs=$2
seq_len=$3
hostfile=$4

cd "$(dirname "$0")"

if [ ! -d "gpt_result" ]; then
	mkdir gpt_result
fi

dir=gpt_result/gbs${gbs}_seq${seq_len}_profile
if [ ! -d $dir ]; then
	mkdir  $dir
fi

dir=gpt_result/gbs${gbs}_seq${seq_len}_evenpart
if [ ! -d $dir ]; then
	mkdir  $dir
fi

dir=gpt_result/gbs${gbs}_seq${seq_len}_adapipe
if [ ! -d $dir ]; then
	mkdir  $dir
fi

for tp in 8 4 2 1
do
	res=$(( $ws / $tp ))
	pp=1
	while [  $pp -le $res ];
	do
		sh gpt_launch_test.sh $ws $tp $pp $gbs $seq_len $hostfile
		pp=$((pp * 2))
	done
done
