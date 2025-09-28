import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI
from lib.datasets.waymo import Waymo
from lib.datasets.nusc_kitti import NUSC_KITTI
from lib.datasets.carla import CARLA
from lib.datasets.coda import CODA

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg):
    # --------------  build kitti dataset -----------------
    if cfg['type'] == 'kitti':
        train_name= 'train' if 'train_split_name' not in cfg.keys() else cfg['train_split_name']
        val_name  = 'val'   if 'val_split_name'   not in cfg.keys() else cfg['val_split_name']
        eval_dataset = 'kitti' if 'eval_dataset' not in cfg.keys() else cfg['eval_dataset']

        train_set = KITTI(root_dir=cfg['root_dir'], split= train_name, cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=4,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = KITTI(root_dir=cfg['root_dir'], split= val_name, cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        # We only use test set of KITTI
        if eval_dataset == "kitti":
            test_set = KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg)
            test_loader = DataLoader(dataset=test_set,
                                     batch_size=cfg['batch_size']//2,
                                     num_workers=2,
                                     worker_init_fn=my_worker_init_fn,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False)
        else:
            test_loader = None
        return train_loader, val_loader, test_loader

    # --------------  build Waymo dataset -----------------
    elif cfg['type'] == 'waymo':
        train_name= 'train' if 'train_split_name' not in cfg.keys() else cfg['train_split_name']
        val_name  = 'val'   if 'val_split_name'   not in cfg.keys() else cfg['val_split_name']

        train_set = Waymo(root_dir=cfg['root_dir'], split= train_name, cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=4,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = Waymo(root_dir=cfg['root_dir'], split=val_name, cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size']//2,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        test_loader = None
        return train_loader, val_loader, test_loader

    # --------------  build nusc_kitti dataset -----------------
    elif cfg['type'] == 'nusc_kitti':
        train_name= 'train' if 'train_split_name' not in cfg.keys() else cfg['train_split_name']
        val_name  = 'val'   if 'val_split_name'   not in cfg.keys() else cfg['val_split_name']

        train_set = NUSC_KITTI(root_dir=cfg['root_dir'], split= train_name, cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=4,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = NUSC_KITTI(root_dir=cfg['root_dir'], split=val_name, cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size']//2,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        test_loader = None
        return train_loader, val_loader, test_loader

    # --------------  build carla dataset -----------------
    elif cfg['type'] == 'carla':
        train_name= 'train' if 'train_split_name' not in cfg.keys() else cfg['train_split_name']
        val_name  = 'val'   if 'val_split_name'   not in cfg.keys() else cfg['val_split_name']
        ext_config = 'pitch0' if 'ext_config'     not in cfg.keys() else cfg['ext_config']

        train_set = CARLA(root_dir=cfg['root_dir'], split= train_name, cfg=cfg, ext_config= ext_config)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=4,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = CARLA(root_dir=cfg['root_dir'], split=val_name, cfg=cfg, ext_config= ext_config)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size']//2,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        test_set = CARLA(root_dir=cfg['root_dir'], split=val_name, cfg=cfg, ext_config= 'height30')
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size']//2,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader

    # --------------  build carla dataset -----------------
    elif cfg['type'] == 'coda':
        train_name= 'train' if 'train_split_name' not in cfg.keys() else cfg['train_split_name']
        val_name  = 'val'   if 'val_split_name'   not in cfg.keys() else cfg['val_split_name']
        eval_dataset = 'coda' if 'eval_dataset' not in cfg.keys() else cfg['eval_dataset']

        train_set = CODA(root_dir=cfg['root_dir'], split= train_name, cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=4,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = CODA(root_dir=cfg['root_dir'], split= val_name, cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        test_set = CODA(root_dir=cfg['root_dir'], split= val_name, cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

