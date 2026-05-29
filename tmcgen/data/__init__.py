from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Batch, Data
from tmcgen.data.featurize import construct_featurizer
from tmcgen.data.dataset import (
    BaseDataset, DockScoreDataset,
)
from torch.utils.data.distributed import DistributedSampler



def get_mode_specific_base_params(args, mode='train'):
    base_params = {}
    if mode == 'train':
        complex_list_file = args.train_complex_list_file
        complex_dir = args.train_complex_dir
        dataset = args.train_dataset
        
        size_sorted = args.train_size_sorted if 'train_size_sorted' in args else False
    
    elif mode == 'val':
        complex_list_file = args.val_complex_list_file \
            if args.val_complex_list_file is not None else args.train_complex_list_file
        complex_dir = args.val_complex_dir \
            if args.val_complex_dir is not None else args.train_complex_dir
        dataset = args.val_dataset \
            if args.val_dataset is not None else args.train_dataset
        

        size_sorted = False            
        if 'val_size_sorted' in args:
            size_sorted = args.val_size_sorted
        else:
            if 'train_size_sorted' in args:
                size_sorted = args.train_size_sorted

    base_params = {
        'complex_list_file': complex_list_file,
        'complex_dir': complex_dir,
        'dataset': dataset,
        'size_sorted': size_sorted
    }

    
    return base_params




def build_score_dataset(args, mode: str = "train"):

    from tmcgen.data.transforms import construct_score_transform

    base_params = get_mode_specific_base_params(args=args, mode=mode)

    if mode in ["val", "test"]:
        timepoints_per_complex = 1
    else:
        timepoints_per_complex = args.timepoints_per_complex

    featurizer = construct_featurizer(args=args)
    transform = construct_score_transform(args, mode='train')

    params = {
        "root": args.data_dir,
        "parser": None,
        "featurizer": featurizer,
        "transform": transform,
        "mode": mode,
        "resolution": args.resolution,
        "agent_type": args.agent_type,
        "center_complex": args.center_complex,
        "esm_embeddings_path": None,
        "timepoints_per_complex": timepoints_per_complex,
        "node_fdim": args.node_fdim,
        "use_rdkit_confs": args.use_rdkit_confs,
    }

    params.update(base_params)
    dataset = DockScoreDataset(**params)
    return dataset
    

def build_data_loader(args, mode: str = "train"):
    if args.model in [
        "dock_score", "dock_score_hetero", "dock_reward", "dock_reward_hetero"
    ]:
        dataset = build_reward_dataset(args=args, mode=mode)
    
    dataset = build_score_dataset(args=args, mode=mode)
   


    if mode == "train":
        batch_size = args.train_bs
    elif mode == "val":
        batch_size = args.val_bs

    if args.n_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
    else:
        sampler = None
    loader_cls = DataLoader
    
    loader = loader_cls(
        dataset=dataset, 
        batch_size=batch_size, 
        collate_fn=custom_collate_fn if  args.model != "confidence" else None, 
        #pin_memory=True,
        drop_last=True,shuffle=(mode == "train") if sampler is None else False, sampler=sampler,
    )
    return loader



def custom_collate_fn(batch):
    
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        raise ValueError("All elements in the batch are None")

    #padded_input, input_lengths = prepare_padded_input(batch)
    
    #return padded_input, input_lengths
    if len(batch) == 0:
        print("[WARNING] Entire batch was None", flush=True)
        return None
    return Batch.from_data_list(batch)

