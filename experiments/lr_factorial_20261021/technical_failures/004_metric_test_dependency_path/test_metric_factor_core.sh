cd /home/futurenetworklab/choihyerim1129/yolo_lightweight_optim || exit 1
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 venv/bin/python review_20261021_lr_factorial/test_metric_factor_core.py --reference review_20261017_independent/evaluation_tools/common_ap_core.py --output review_20261021_lr_factorial/metric_factor_core_validation.json
