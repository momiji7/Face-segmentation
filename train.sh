CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=6 train.py \
        --train_lists /search/speech/xz/datasets/segment/CelebAMask-HQ/CelebAMask-HQ.json \
        --save_path ./snapshots/  \
        --input_height 512 \
        --input_width 512 \
        --scale_lists 0.75 1.0 1.5 1.75 2.0 \
        --batch_size_per_gpu  8 \
        --nclass 19 \
        --LR 0.01 \
        --epochs 50 \
        --gpu_num 6
        
        
    