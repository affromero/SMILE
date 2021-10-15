# import torch.utils.data.distributed
from misc.utils import to_cuda
from misc.utils import copy_file_to_screen
import warnings
import torch
import copy
import os
from misc.solver_utils import Solver_Utils
from munch import Munch

from data_loader import get_loader
warnings.filterwarnings('ignore')


class Solver(Solver_Utils):
    def __init__(self, args, data_loader):
        # Data loader
        self.data_loader = data_loader
        if args.mode == 'train':
            self.data_loader_val = get_loader(args,
                                              all_attr=args.ALL_ATTR,
                                              shuffling=True,
                                              mode='val')
        self.args = args
        self.args.data_module = self.data_loader.dataset
        self.args.domains = data_loader.dataset.selected_attrs
        self.args.n_domains = len(data_loader.dataset.selected_attrs)
        self.args.c_dim = self.args.n_domains
        self.dist = args.dist
        self.verbose = 1 if self.dist.rank() == 0 else 0
        self.args.num_domains = len(self.args.data_module.parent_attrs.keys())

        # if self.verbose:
        self.build_model()
        self.build_solver()

        if args.mode == 'train' and args.GPU[0] != '-1' and self.dist.rank(
        ) == 0:
            copy_file_to_screen(args.sample_path, args.GPU)

        self.dist.barrier()

    # ==================================================================#
    # ==================================================================#
    def build_model(self):
        from models.generator import Generator
        from models.discriminator import Discriminator
        from models.mapping import Noise2Style
        from models.encoder import StyleEncoder

        if self.args.TRAIN_MASK:
            in_dim = self.args.mask_dim
        else:
            in_dim = self.args.color_dim

        debug = self.args.mode == 'train' and self.verbose

        discriminator = Discriminator(self.args, color_dim=in_dim, debug=debug)
        style_encoder = StyleEncoder(self.args, color_dim=in_dim, debug=debug)
        mapping_network = Noise2Style(self.args, debug=debug)
        generator = Generator(self.args, color_dim=in_dim, debug=debug)

        self.nets = Munch(
            G=generator,
            F=mapping_network,
            S=style_encoder,
            D=discriminator,
        )

        if debug:
            self.print_network(self.nets.D, 'Discriminator')
            self.print_network(self.nets.S, 'Style Encoder')
            self.print_network(self.nets.F, 'Mapping')
            self.print_network(self.nets.G, 'Generator')

    def build_solver(self):
        if self.args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=self.args.f_lr if 'F' in net else self.args.lr,
                    betas=[self.args.beta1, self.args.beta2],
                    weight_decay=self.args.weight_decay)

                if self.dist.size() > 1 and self.dist.horovod:
                    # Horovod: broadcast parameters & optimizer state.
                    self.dist.broadcast_parameters(self.nets[net].state_dict(),
                                                   root_rank=0)
                    self.dist.broadcast_optimizer_state(self.optims[net],
                                                        root_rank=0)
                    self.optims[net] = self.dist.DistributedOptimizer(
                        self.optims[net],
                        named_parameters=self.nets[net].named_parameters(),
                        op=self.dist.hvd.Average)

        self.nets_ema = copy.deepcopy(self.nets)

        # Start with trained model
        if self.args.pretrained_model:
            self.load_pretrained_model()

        self._to_cuda()

        if self.args.FAN and self.args.dataset != 'DeepFashion2':
            from misc.wing import FAN
            general_attr = self.args.GENERAL_HEATMAP
            self.nets.FAN = FAN(
                fname_pretrained='models/pretrained_models/wing.ckpt',
                general_attr=general_attr)
            self.nets.FAN = to_cuda(self.nets.FAN, fixed=True)
            self.nets_ema.FAN = self.nets.FAN

        n_domains = list(self.args.data_module.selected_attrs)
        _sr = '{} domains involved: {}'.format(len(n_domains), str(n_domains))

        self.PRINT(_sr)

    # ============================================================#
    # ============================================================#
    def update_lr(self, lr, f_lr):
        for key in self.optims.keys():
            if 'F' in key:
                for param_group in self.optims[key].param_groups:
                    param_group['lr'] = f_lr
            else:
                for param_group in self.optims[key].param_groups:
                    param_group['lr'] = lr

    # ============================================================#
    # ============================================================#
    def reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    # ==================================================================#
    # ==================================================================#
    def _to_cuda(self):
        for key in self.nets.keys():
            self.nets[key] = to_cuda(self.nets[key], fixed=self.args.HOROVOD)

        for key in self.nets_ema.keys():
            # self.nets_ema[key] = to_cuda(self.nets_ema[key], fixed=True)
            self.nets_ema[key] = to_cuda(self.nets_ema[key],
                                         fixed=self.args.HOROVOD)
