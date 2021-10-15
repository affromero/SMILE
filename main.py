#!/usr/bin/python
# Distributed: python -m torch.distributed.launch --nproc_per_node=$n
# main.py **kwargs
import os
from data_loader import get_loader
import config as cfg
import warnings
import sys
import torch
import torch.distributed
from misc.utils import distributed, distributed_horovod
from misc.utils import PRINT, config_yaml
warnings.filterwarnings('ignore')
# torch.autograd.set_detect_anomaly(True)


def set_score(config, *kwargs):
    if config.EVAL:
        from test import Test
        scores = Test(config, *kwargs)
        scores.Eval()
        return True
    return False


def _PRINT(config):
    string = '------------ Options -------------'
    PRINT(config.log, string)
    for k, v in sorted(vars(config).items()):
        string = '%s: %s' % (str(k), str(v))
        PRINT(config.log, string)
    string = '-------------- End ---------------'
    PRINT(config.log, string)


def main(config):
    from torch.backends import cudnn
    # For fast training
    cudnn.benchmark = True

    data_loader = get_loader(
        config,
        all_attr=config.ALL_ATTR,
        sampled=config.DATASET_SAMPLED,
        verbose=config.mode == 'train' and config.dist.rank() == 0,
    )

    if set_score(config, data_loader):
        return

    if config.mode == 'train':
        from train import Train
        Train(config, data_loader)
        # from test import Test
        # test = Test(config, data_loader)
        # test(dataset=config.dataset_test)

    elif config.mode == 'test':
        from test import Test
        test = Test(config, data_loader)
        if config.DEMO_PATH:
            test.DEMO(config.DEMO_PATH)
        else:
            test.sample(dataset=config.dataset_test)

    elif config.mode == 'demo':
        from demo import Demo
        # from demo_scientifica import Demo
        Demo(config, data_loader)()


if __name__ == '__main__':
    from misc.options import base_parser
    config = base_parser()
    os.environ['TORCH_EXTENSIONS_DIR'] = '__pytorch__'
    # os.environ['QT_DEBUG_PLUGINS']= '1'
    if config.HOROVOD:
        config.dist = distributed_horovod()
        config.dist.init()
        # Horovod
        # torch.cuda.set_device(config.dist.local_rank())
        # config.GPU = [int(i) for i in range(dist.size())]
        config.GPU = [int(i) for i in config.GPU.split(',')]
        gpu_rank = config.dist.rank()
        # print(gpu_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU[gpu_rank])
        config.lr *= config.dist.size()
        config.f_lr *= config.dist.size()

    else:
        if config.GPU == 'NO_CUDA':
            config.GPU = '-1'
        if len(config.GPU.split(',')) > 1:
            if config.DISTRIBUTED:
                config.GPU = [int(i) for i in config.GPU.split(',')]
                gpu_rank = config.local_rank
                # print(gpu_rank)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU[gpu_rank])
                world_size = len(config.GPU)
                # torch.cuda.set_device(gpu_rank)
                torch.distributed.init_process_group(backend='nccl',
                                                     world_size=world_size,
                                                     rank=gpu_rank)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
                config.GPU = [int(i) for i in config.GPU.split(',')]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
            config.GPU = [int(i) for i in config.GPU.split(',')]
        config.dist = distributed()

    if config.mode == 'demo':
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    config_yaml(config, 'datasets/{}.yaml'.format(config.dataset))
    config.TENSORBOARD = config.TENSORBOARD and config.GPU != ['-1']
    config.VISDOM = config.VISDOM and config.GPU != ['-1']
    config = cfg.update_config(config)
    if config.mode == 'train' and config.GPU != ['-1']:
        if config.dist.rank() == 0:
            PRINT(config.log, ' '.join(sys.argv))
            _PRINT(config)
        main(config)
        config.log.close()

    else:
        main(config)
