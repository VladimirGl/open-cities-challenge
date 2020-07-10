python -m src.predict \
    --configs $(ls configs/stage1-eye*) \
    --test_dir participant_data/train_0/ \
    --test_csv participant_data/train_0/test.csv \
    --dst_dir participant_data/train_0/eye/ \
    --batch_size 8 \
    --gpu '0'