CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=6 train.py \
        --dataset_name cityscapes \
        --train_lists /search/speech/xz/datasets/segment/cityscapes/cityscapes_train.json \
        --eval_lists /search/speech/xz/datasets/segment/cityscapes/cityscapes_eval.json \
        --save_path ./snapshots/  \
        --input_height 1024 \
        --input_width 1024 \
        --scale_lists 0.75 1.0 1.5 1.75 2.0 \
        --batch_size_per_gpu  4 \
        --nclass 19 \
        --LR 0.01 \
        --epochs 50 \
        --gpu_num 6 \
        --use_fp16
        
        
#--dataset_name CelebAMask-HQ \
#--train_lists /search/speech/xz/datasets/segment/CelebAMask-HQ/CelebAMask-HQ.json \
        
    