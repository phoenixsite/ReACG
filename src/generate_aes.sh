#!/bin/bash

RESULTS_DIR=../result/
PROGRAM=run_evaluation.py
BATCH_SIZE=150
LOG_LEVEL=30
NTHREADS=4

for config_file in ../configs/transferability/acg*.yaml
do
	if [ $config_file == "../configs/transferability/acg_resnet50_eps1.yaml" ] || [ $config_file == "../configs/transferability/acg_vgg16_eps1.yaml" ] || [ $config_file == "../configs/transferability/acg_vgg16_eps2.yaml" ]; then
		echo "Executing ACG with $config_file"
		uv run $PROGRAM -p $config_file \
			-g 0 \
			-o $RESULTS_DIR \
			--log_level $LOG_LEVEL \
			--save-adversarial \
			--n_threads $NTHREADS \
			--cmd_param batch_size:int:$BATCH_SIZE
	fi
done

#for config_file in ../configs/transferability/apgd*.yaml
#do
#	echo  "Executing APGD with $config_file"
#	uv run $PROGRAM -p $config_file \
#		-g 0 \
#		-o $RESULTS_DIR \
#		--log_level $LOG_LEVEL \
#		--save-adversarial \
#		--n_threads $NTHREADS \
#		--cmd_param batch_size:int:$BATCH_SIZE
#done

for config_file in ../configs/transferability/reacg*.yaml
do
	if [ $config_file == "../configs/transferability/reacg_resnet50_eps1.yaml" ] || [ $config_file == "../configs/transferability/reacg_vgg16_eps1.yaml" ] || [ $config_file == "../configs/transferability/reacg_vgg16_eps2.yaml" ]; then
		echo "Executing ReACG with $config_file"
		uv run $PROGRAM -p $config_file \
			-g 0 \
			-o $RESULTS_DIR \
			--log_level $LOG_LEVEL \
			--save-adversarial \
			--n_threads $NTHREADS \
			--cmd_param batch_size:int:$BATCH_SIZE
	fi
done