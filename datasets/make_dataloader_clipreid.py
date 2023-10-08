import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
# from .dukemtmcreid import DukeMTMCreID
# from .market1501 import Market1501
# from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
# from .occ_duke import OCC_DukeMTMCreID
# from .vehicleid import VehicleID
# from .veri import VeRi
from .morph import Morph
from .history import History
from .crowdbeauty import CrowdBeauty
from .adience import Adience
from .fgnet import Fgnet
from .clap import Clap
from .cacd import Cacd
from .morph_S3 import MorphS3
from .morph_S4 import MorphS4
from .morph_S4_val import MorphS4_val

__factory = {
    'morph': Morph,
    'history': History,
    'crowdbeauty': CrowdBeauty,
    'adience': Adience,
    'fgnet': Fgnet,
    'clap': Clap,
    'cacd': Cacd,      
    'morph_s3': MorphS3,
    'morph_s4': MorphS4,
    'morph_s4_val': MorphS4_val
}

def train_collate_fn(batch):
    imgs, ages = zip(*batch)
    ages = torch.tensor(ages, dtype=torch.int64)
    return torch.stack(imgs, dim=0), ages

def val_collate_fn(batch):
    imgs, ages = zip(*batch)
    ages = torch.tensor(ages, dtype=torch.int64)
    return torch.stack(imgs, dim=0), ages

def make_dataloader(cfg, i=None):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, train_txt_file=cfg.DATASETS.TRAIN_TXT_FILES, test_txt_file=cfg.DATASETS.TEST_TXT_FILES, idx=i)
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_ages
    # cam_num = dataset.num_train_cams
    # view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                shuffle=True,
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.val, val_transforms)
    test_set = ImageDataset(dataset.test, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader_stage2, train_loader_stage1, val_loader, test_loader, len(dataset.test), num_classes#, #pid2label#, view_num
