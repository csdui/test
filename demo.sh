#!/bin/bash
python example.py \
--arch REDNet30 \
--weights_path ./model_saved/REDNet30_epoch_last.pth \
--image_path ./data/monarch_REDNet30.png \
--outputs_dir ./produced \
--jpeg_quality 10