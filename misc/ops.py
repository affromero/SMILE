import os


class ops(object):
    # ============================================================#
    # ============================================================#
    def add_args(self, args):
        self.args = args

    # ============================================================#
    # ============================================================#
    def add_dataloader(self, data_loader):
        self.data_loader = data_loader

    # ==================================================================#
    # ==================================================================#
    @property
    def MultiLabel_Datasets(self):
        return [
            'BP4D', 'CelebA', 'CelebA_HQ', 'CelebA_AFHQ', 'AFFHQ', 'FFHQ'
            'EmotioNet', 'DeepFashion2', 'DEMO'
        ]

    # ==================================================================#
    # ==================================================================#
    def OneDomainDatasets(self):
        return ['painters_14']

    # ==================================================================#
    # ==================================================================#
    def get_beings(self):
        if self.data_loader.dataset.name in ['CelebA_AFHQ', 'AFFHQ']:
            living_beings = ['Male', 'Female', 'Cat', 'Dog', 'Wild']
        else:
            living_beings = ['Male', 'Female']
        return living_beings

    # ==================================================================#
    # ==================================================================#
    def target_multiAttr(self, target, index):
        target = target.clone()
        all_attr = list(self.data_loader.dataset.selected_attrs)
        if self.data_loader.dataset.args.USE_MASK_TRAIN_ATTR:
            all_attr += list(self.data_loader.dataset.semantic_attr.keys())
        attr2idx = self.data_loader.dataset.attr2idx

        def replace(target, attrs):
            for possible_attr in attrs:
                if possible_attr not in all_attr:
                    return target
            if all_attr[index] in attrs:
                for attr in attrs:
                    if attr in all_attr:
                        target[:, attr2idx[attr]] = 0
                target[:, index] = 1
            return target

        # color_hair = [
        #     'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
        # ]
        # style_hair = ['Bald', 'Straight_Hair', 'Wavy_Hair']
        # target = replace(target, color_hair)
        # target = replace(target, style_hair)
        # target = replace(target, ['Bald', 'Bangs'])
        for parents in self.data_loader.dataset.parent_attrs.values():
            if len(parents) > 1:
                target = replace(target, parents)
        # target = replace(target, self.get_beings())
        # target = replace(target, ['Young', 'Aged'])
        # target = replace(target, ['Eyeglasses', 'NOT_Eyeglasses'])
        # target = replace(target, ['Earrings', 'NOT_Earrings'])
        # target = replace(target, ['Hat', 'NOT_Hat'])
        # target = replace(target, ['Bangs', 'NOT_Bangs'])
        # target = replace(target, ['Hair', 'NOT_Hair'])
        # target = replace(target, ['Smiling', 'NOT_Smiling'])
        # target = replace(target, ['short_sleeve_top', 'long_sleeve_top'])
        # target = replace(target, ['vest', 'shorts', 'trousers', 'skirt'])
        return target

    # ==================================================================#
    # ==================================================================#
    def remove_redundancyAttr(self, target):
        import random
        selected_attrs = list(self.data_loader.dataset.selected_attrs)
        parent_attr = dict(self.data_loader.dataset.parent_attrs)

        def replace(attrs, target):
            if attrs[0] not in selected_attrs:
                return target
            idx = random.randint(0, len(attrs) - 1)
            init_idx = selected_attrs.index(attrs[0])
            for i in range(len(attrs)):
                target[init_idx + i] = 0
            target[init_idx + idx] = 1

        if self.args.BINARY_CHILDREN:
            # at least one attribute must be shared
            # eg, beings: male and female.
            for i in range(target.shape[0]):
                for attrs in parent_attr.values():
                    if len(attrs) > 1:
                        replace(attrs, target[i])

        return target

    # ==================================================================#
    # ==================================================================#
    def PRINT(self, str):
        from misc.utils import PRINT
        if self.verbose:
            if self.args.mode == 'train':
                PRINT(self.args.log, str)
            else:
                print(str)

    # ==================================================================#
    # ==================================================================#
    def PLOT(self, Epoch):
        import numpy as np
        from misc.utils import plot_txt
        LOSS = {
            key: np.array(value).mean()
            for key, value in self.LOSS.items()
        }
        if not os.path.isfile(self.args.loss_plot):
            with open(self.args.loss_plot, 'w') as f:
                f.writelines('{}\n'.format('\t'.join(['Epoch'] +
                                                     list(LOSS.keys()))))
        with open(self.args.loss_plot, 'a') as f:
            f.writelines('{}\n'.format(
                '\t'.join([str(Epoch)] + [str(i)
                                          for i in list(LOSS.values())])))
        plot_txt(self.args.loss_plot)

    # ==================================================================#
    # ==================================================================#
    def PRINT_LOG(self, batch_size, _print=True):
        from termcolor import colored
        # import GPUtil
        # import humanize
        # 'grey', 'red', 'green', 'yellow', 'blue', 'magenta, 'cyan', 'white'
        import socket
        _gpu = 'GPU: {}'.format(self.args.GPU)
        if hasattr(self, 'GPU_MEMORY_USED'):
            gpu_memory = self.GPU_MEMORY_USED
            _gpu += ' {}MB'.format(gpu_memory)
        elif self.args.GPU[0] != -1:
            gpu_memory = self.get_gpu_memory_used()
            _gpu += ' {}MB'.format(gpu_memory)
        gpu_string = colored(_gpu, 'green')
        Log = "---> bs: {}, img: {}, {} |".format(batch_size,
                                                  self.args.image_size,
                                                  gpu_string)

        for k, v in sorted(vars(self.args).items()):
            if isinstance(v, bool) and v:
                if k in ['DELETE']:
                    continue
                Log += ' [*{}]'.format(k)
        dataset_string = '{} [{}%]'.format(self.args.dataset,
                                           self.args.DATASET_SAMPLED)
        dataset_string = colored(dataset_string, 'red')
        Log += ' [*{}]'.format(dataset_string)
        pid_string = 'pid: {}'.format(os.getpid())
        Log += ' [{}]'.format(colored(pid_string, 'magenta'))
        if 'JOB_ID' in os.environ.keys():
            job_string = 'job-ID: {}'.format(os.environ['JOB_ID'])
            Log += ' [{}]'.format(colored(job_string, 'cyan'))
            queue_string = '{}'.format(os.environ['QUEUE'])
            Log += ' [{}]'.format(colored(queue_string, 'blue'))
        cluster_string = colored(socket.gethostname(), 'yellow')
        Log += ' [-> {} <-]'.format(cluster_string)
        if _print:
            self.PRINT(Log)
        return Log

    # ==================================================================#
    # ==================================================================#
    def get_labels(self,
                   style_guided=False,
                   from_content=False,
                   semantics=False,
                   TOTAL=False):
        from misc.utils import get_labels
        return get_labels(self.data_loader.dataset.image_size,
                          self.data_loader.dataset.name,
                          style_guided=style_guided,
                          binary=self.args.ATTR != '',
                          from_content=from_content,
                          semantics=semantics,
                          data_info=self.data_loader.dataset,
                          total_column=TOTAL)
