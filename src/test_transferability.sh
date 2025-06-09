#!/bin/bash


RESULT_DIR=../result_transfer
LOG_LEVEL=30
BATCH_SIZE=30
PROGRAM=run_transferability.py

# for test_dir in ../result/2025-06-07/*
# do
#         aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
#         echo "Using $aes_dir directory"
#         for target_model in "inception_resnet_v2.tf_ens_adv_in1k" "vgg16.tv_in1k" "inception_v3.tv_in1k" "resnet50.tv2_in1k" "inception_v3.tf_adv_in1k" "vgg19.tv_in1k" "resnet152.tv2_in1k" "mobilenetv2_140.ra_in1k"
#         do
#                 echo "Testing model: $target_model"
#                 uv run "$PROGRAM" \
#                         -d "$aes_dir" \
#                         -tm "$target_model" \
#                         --log-level "$LOG_LEVEL" \
#                         -o "$RESULT_DIR/$target_model" \
#                         -g 0 \
#                         --test-transferability \
#                         --test-samples \
#                         -bs "$BATCH_SIZE"
#         done
# done

for test_dir in ../result/2025-06-07/*
do
        aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
        echo "Using $aes_dir directory"
        for target_model in "inception_resnet_v2.tf_ens_adv_in1k" "vgg16.tv_in1k" "inception_v3.tv_in1k" "resnet50.tv2_in1k" "inception_v3.tf_adv_in1k" "vgg19.tv_in1k" "resnet152.tv2_in1k" "mobilenetv2_140.ra_in1k"
        do
                echo "Testing model: $target_model"
                uv run "$PROGRAM" \
                        -d "$aes_dir" \
                        -tm "$target_model" \
                        --log-level "$LOG_LEVEL" \
                        -o "$RESULT_DIR/$target_model" \
                        -g 0 \
                        --test-samples \
                        -bs "$BATCH_SIZE"
        done
done