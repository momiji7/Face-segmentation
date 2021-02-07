from dataset.datasets import MaskDataset
from dataset.cityscapes import CityScapesDataset
import dataset.transforms as transforms
import torch 


def get_dataset(args, mode = 'train'):
    
    if args.dataset_name == 'CelebAMask-HQ':
        if mode == 'train':
            mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
            transform_list = transforms.Compose([
                transforms.TrainScale2WH((args.input_width, args.input_height)),
                transforms.AugHorizontalFlip(args.flip_prob),
                transforms.AugScale(args.scale_lists),
                transforms.AugCrop(args.input_width, args.input_height, args.crop_perturb_max, mean_fill),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])            
            dataset = MaskDataset(args, transform_list, args.train_lists)
            
    elif args.dataset_name == 'cityscapes':
        if mode == 'train':
            mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
            transform_list = transforms.Compose([
                transforms.TrainScale2WH((args.input_width, args.input_height)),
                transforms.AugHorizontalFlip(args.flip_prob),
                transforms.AugScale(args.scale_lists),
                transforms.AugCrop(args.input_width, args.input_height, args.crop_perturb_max, mean_fill),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115])
            ])            
            dataset = CityScapesDataset(args, transform_list, args.train_lists)
        else:
            transform_list = transforms.Compose([
                transforms.TrainScale2WH((args.input_width, args.input_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3223, 0.3690, 0.3257], std=[0.2115, 0.2148, 0.2112])
            ])            
            dataset = CityScapesDataset(args, transform_list, args.eval_lists)
    else:
        raise
            
    return dataset

def get_dataloader(args, mode = 'train'):
    
    if mode == 'train':
        dataset = get_dataset(args, mode)
        sampler = None 
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle= sampler == None, sampler = sampler, num_workers= 8, pin_memory=True, drop_last = True)
        return data_loader
    else:
        dataset = get_dataset(args, mode)
        sampler = None 
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle= False, sampler = sampler, num_workers= 8, pin_memory=True, drop_last = False)
        return data_loader