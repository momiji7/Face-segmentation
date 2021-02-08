import os, sys, time, random, argparse


def obtain_basic_args():
    parser = argparse.ArgumentParser(description='Segmentaion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset_name',     type=str,                  )
    parser.add_argument('--train_lists',      type=str,   nargs='+',      help='The list file path to the video training dataset.')
    parser.add_argument('--eval_lists',       type=str,   nargs='+',      help='The list file path to the image testing dataset.')
    parser.add_argument('--nclass',           type=int,                   help='The number of segmentation classes')
    parser.add_argument('--gpu_num',          type=int,   default=6,      help='The number of used gpu')
    parser.add_argument('--no_apex',  dest='no_apex', action='store_true', help='Donot use apex')
    parser.add_argument('--use_fp16', dest='use_fp16', action='store_true')

  # Data Transform
    parser.add_argument('--input_height',     type=int,   default=256   )
    parser.add_argument('--input_width',      type=int,   default=256   )
    parser.add_argument('--flip_prob',        type=float, default=0.5,    help='argument flip probability.')
    parser.add_argument('--crop_perturb_max', type=int,   default=5,     help='argument crop : center of maximum perturb distance.')
    parser.add_argument('--scale_lists',      type=float,   nargs='+',      help='The list file path to the video training dataset.')

    parser.add_argument('--batch_size_per_gpu',type=int,   default=8,      help='Batch size per gpu for training.')
    # Checkpoints
    parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
    parser.add_argument('--save_path',        type=str,                   help='Folder to save checkpoints and log.')



    # Optimizer
    parser.add_argument('--optimizer_type',   type=str,   default='sgd',  help='Optimizer type')
    parser.add_argument('--LR',               type=float, default=0.025,  help='Learning rate for optimizer.')
    parser.add_argument('--momentum',         type=float, default=0.9,    help='Momentum for optimizer.')
    parser.add_argument('--decay',            type=float, default=0.0001, help='Decay for optimizer.')
    parser.add_argument('--nesterov',         action='store_true',        help='Using nesterov for optimizer.')
    parser.add_argument('--epochs',           type=int,   default=100,    help='Epochs for training')
    parser.add_argument('--loss_alpha',       type=float, default=1.0,    )

    # lr_scheduler
    parser.add_argument('--scheduler_type',   type=str,   default='LambdaLR',  help='Schedule type')
    parser.add_argument('--scheduler_power',  type=float, default=0.9,    help='Decay for learning rate.')

    # model
    parser.add_argument('--model_type',       type=str,   default='BiseNet',  help='Model type')

    # distributed training
    parser.add_argument('--use_distributed',  action='store_true',        help='If use distributed or not')
    parser.add_argument('--local_rank',       type=int,   default=0,      help='Necessarily distributed arguments')


    args = parser.parse_args()
    
    return args