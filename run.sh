cd mlperf_training/object_detection/pytorch
python setup.py develop
pip install opencv-contrib-python==4.5.5.64

git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

python tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025
