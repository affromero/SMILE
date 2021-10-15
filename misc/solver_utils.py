import torch
import os
from misc.utils import to_cuda, to_data
from misc.utils import single_source
from misc.utils import create_arrow, create_circle, denorm
from misc.utils import color_frame, human_format, compress_image
from torchvision.utils import save_image
from glob import glob
from misc.ops import ops
from misc.mask_utils import label2mask, mask2label


class Solver_Utils(ops):
    # ==================================================================#
    # ==================================================================#
    def print_network(self, model, name):
        num_params = 0
        num_learns = 0
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                num_learns += p.numel()
        self.PRINT(
            "{} number of parameters (TOTAL): {}\t(LEARNABLE): {}.".format(
                name.upper(), human_format(num_params),
                human_format(num_learns)))
        # self.PRINT(name)
        # self.PRINT(model)

    # ============================================================#
    # ============================================================#
    def output_sample(self, epoch, iter):
        return os.path.join(
            self.args.sample_path, '{}_{}.jpg'.format(
                str(epoch).zfill(4),
                str(iter).zfill(len(str(len(self.data_loader))))))

    # ============================================================#
    # ============================================================#
    def output_model(self, epoch, iter):
        return os.path.join(
            self.args.model_save_path, '{}_{}_{}.pth'.format(
                str(epoch).zfill(4),
                str(iter).zfill(len(str(len(self.data_loader)))), '{}'))

    # ==================================================================#
    # ==================================================================#
    def save(self, Epoch, iter):
        name = self.output_model(Epoch, iter)
        dirname = os.path.dirname(name)
        namefile = os.path.basename(name).split('_')[0]
        all_files = sorted(glob(os.path.join(dirname,
                                             namefile[:-1] + '*.pth')))
        for line in all_files:
            _epoch = int(os.path.basename(line).split('_')[0]) + 1
            if (_epoch % self.args.model_epoch) == 0:
                continue
            else:
                os.remove(line)
        for key, value in self.nets.items():
            torch.save(value.state_dict(), name.format(key))
        for key, value in self.nets_ema.items():
            torch.save(value.state_dict(), name.format(key + '_ema'))

    # ==================================================================#
    # ==================================================================#
    def load_pretrained_model(self):
        self.PRINT('Resuming model (step: {})...'.format(
            self.args.pretrained_model))
        # self.name = os.path.join(
        #     self.args.model_save_path,
        #     '{}_{}.pth'.format(self.args.pretrained_model, '{}'))
        self.name = os.path.join('{}_{}.pth'.format(self.args.pretrained_model,
                                                    '{}'))
        self.PRINT('Model: {}'.format(self.name))

        def load(model, name='G'):
            if 'D' in key and self.args.mode != 'train':
                return
            weights = torch.load(self.name.format(name),
                                 map_location=lambda storage, loc: storage)
            model.load_state_dict(weights)

        if self.args.mode == 'train':
            for key in self.nets.keys():
                load(self.nets[key], key)
        if self.dist.rank() == 0:
            for key in self.nets_ema.keys():
                load(self.nets_ema[key], key + '_ema')

        print("-- Loading Success!!")

    # ==================================================================#
    # ==================================================================#
    def last_checkpoint(self, model_path=None, name='G_ema'):
        if model_path is None:
            model_path = self.args.model_save_path
        last_file = sorted(glob(os.path.join(model_path, f'*_{name}.pth')))[-1]
        return last_file

    # ==================================================================#
    # ==================================================================#
    def resume_name(self, model_path=None):
        if model_path is None:
            model_path = self.args.model_save_path
        if self.args.pretrained_model in ['', None]:
            try:
                last_file = self.last_checkpoint(model_path)
            except IndexError:
                raise IndexError("No model found at " + model_path)
            last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
        else:
            last_name = self.args.pretrained_model
        return last_name

    # ==================================================================#
    # ==================================================================#
    def get_gpu_memory_used(self):
        import GPUtil
        if torch.cuda.is_available():
            try:
                mem = int(GPUtil.getGPUs()[self.args.GPU[0]].memoryUsed)
            except BaseException:
                mem = 0
            if hasattr(self, 'GPU_MEMORY_USED'):
                mem = max(mem, self.GPU_MEMORY_USED)
            return mem
        else:
            return 0

    # ==================================================================#
    # ==================================================================#
    def mask2label(self, seg):
        masklabel = mask2label(seg)
        sem_attr_all = self.args.data_module.mask_label
        sem_attr = self.args.data_module.semantic_attr.keys()
        indexes = [sem_attr_all[i.lower()] for i in sem_attr]
        return masklabel[:, indexes]
