#!/bin/bash
python main.py \
--arch "REDNet30" \
--images_dir ./data_image \
--outputs_dir ./model_saved \
--jpeg_quality 10 \
--patch_size 50 \
--batch_size 16 \
--num_epochs 20 \
--lr 1e-4 \
--threads 8 \
--seed 123 \
--num_epochs 200