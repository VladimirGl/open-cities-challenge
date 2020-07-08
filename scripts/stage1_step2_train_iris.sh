# training 10 models for first stage (LB ~0.8497)

# 5 folds Unet model with EfficientNet-B1 encoder
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-effb1-f0.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-effb1-f1.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-effb1-f2.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-effb1-f3.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-effb1-f4.yaml

# 5 folds Unet model with SE-ResNeXt-32x4d encoder
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-srx50-f0.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-srx50-f1.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-srx50-f2.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-srx50-f3.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config=configs/stage1-iris-srx50-f4.yaml