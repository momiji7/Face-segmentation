from dataset.datasets import MaskDataset
import dataset.transforms as transforms
import torch 


def get_dataloader(args, mode = 'train'):
    
    if mode == 'train':
        mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
        train_transform = transforms.Compose([
            transforms.TrainScale2WH((args.input_width, args.input_height)),
            transforms.AugHorizontalFlip(args.flip_prob),
            transforms.AugScale(args.scale_lists),
            transforms.AugCrop(args.input_width, args.input_height, args.crop_perturb_max, mean_fill),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = MaskDataset(args, train_transform, args.train_lists)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        #sampler = None
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle=False, sampler = sampler, num_workers= 8, pin_memory=True, drop_last = True)
        return data_loader
    elif mode == 'test':
        return None