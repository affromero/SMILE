from metrics.f1_score import compute_f1, plot_PR
import pickle
import torch
import os
import warnings
import time
from tqdm import tqdm
from misc.utils import elapsed_time
from data_loader import get_loader
from solver import Solver
from misc.score_utils import Extract, plot_PR_from_munch
warnings.filterwarnings('ignore')


class PRINT(object):
    def assign_file(self, file):
        self.file = open(file, 'w')

    def __call__(self, text):
        from misc.utils import PRINT
        PRINT(self.file, text)

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.file.close()


@torch.no_grad()
def FID(path, cuda=False, domain=0, verbose=False):
    from metrics.fid_score import calculate_fid_given_paths
    batch_size = 32
    # batch_size = 16
    fid_value = calculate_fid_given_paths(path, batch_size, cuda, 2048)
    if verbose:
        print('\n[Domain: {}] FID: {:.3f}'.format(domain, fid_value))
    return fid_value


@torch.no_grad()
def prdc(path, nearest_k=5, verbose=True):
    from prdc import compute_prdc
    import numpy as np
    assert len(path) == 2
    act0 = np.load(path[0] + '.npz')['act']
    act1 = np.load(path[1] + '.npz')['act']
    metrics = compute_prdc(real_features=act0,
                           fake_features=act1,
                           nearest_k=nearest_k,
                           verbose=verbose)
    return metrics


@torch.no_grad()
class Scores(Solver):
    def __init__(self,
                 config,
                 generator,
                 style_model,
                 mapping,
                 verbose=False,
                 FAN=None,
                 mode='test'):
        self.data_loader = get_loader(
            config,
            mode=mode,
            batch_size=1,
            verbose=verbose and mode == 'test',
        )
        if mode == 'test':
            self.data_loader_train = get_loader(
                config,
                mode='train',
                batch_size=1,
                verbose=verbose,
            )
        else:
            self.data_loader_train = None
        self.MODE = mode
        self.CUDA = torch.cuda.is_available()
        self.generator = generator
        self.style_encoder = style_model
        self.mapping = mapping
        self.FAN = FAN
        self.mask = config.TRAIN_MASK
        self.dist = config.dist
        self.PRINT = PRINT()
        self.verbose = config.mode == 'test' or verbose
        self.config = config

    # ==================================================================#
    # ==================================================================#

    def Eval(self,
             name=None,
             first=False,
             image_guided=False,
             latent_guided=False):
        if not self.config.pretrained_model and name is None:
            print("Not *.pth file in {}. Calculating from scratch...".format(
                self.config.model_save_path))
            self.name = os.path.join(self.config.sample_path, '0_0')
        elif name is None:
            self.name = os.path.join(self.config.sample_path,
                                     self.config.pretrained_model)
        else:
            self.name = name
        self.name += '_' + self.MODE
        _str = "Must choose one guidance [reference, latent] for Eval"
        assert image_guided != latent_guided, _str
        guide_transform = []
        guide_transform += ['reference'] if image_guided else []
        guide_transform += ['latent'] if latent_guided else []
        assert len(
            guide_transform) == 1, 'Please select either reference or latent'
        self.verbose = first or self.config.mode == 'test'
        Time = time.time()
        torch.manual_seed(1)
        if self.CUDA:
            torch.cuda.manual_seed(1)

        data_loader = self.data_loader
        style_fixed = False
        if self.config.mode == 'test':
            fid_calc = ['train', 'fake']
        else:
            fid_calc = ['real', 'fake']
        attrs = data_loader.dataset.selected_attrs
        num_labels = len(attrs)
        guide = guide_transform[0]
        style_str = 'fixed' if style_fixed or guide == 'reference' else 'random'
        file_name = self.name + '_eval_{}_{}.txt'.format(guide, style_str)
        dir_name = self.name + '_eval_{}_{}_style'.format(guide, style_str)

        self.PRINT.assign_file(file_name)

        extract = Extract(guide,
                          data_loader,
                          self.generator,
                          self.style_encoder,
                          self.mapping,
                          dir_name,
                          MODE=self.MODE,
                          data_loader_train=self.data_loader_train,
                          FAN=self.FAN,
                          mask=self.mask,
                          verbose=self.verbose)
        folders = extract.create_folders()

        scores_file = os.path.join(dir_name, 'scores_{}.pkl'.format(guide))
        if os.path.isfile(scores_file):
            partial_scores = pickle.load(
                open(os.path.join(dir_name, 'scores_{}.pkl'.format(guide)),
                     'rb'))
            # Replace binary GENDER with independent Male and Female labels
            parents_attr = list(data_loader.dataset.parent_attrs.keys())[1:]
            parents_attr = ['Male', 'Female'] + parents_attr
            if self.mask:
                # TODO remove this line
                partial_scores.attr_mask['real'] = partial_scores.attr['real']
            partial_scores.attr = compute_f1(partial_scores.attr, parents_attr)
            plot_PR_from_munch(partial_scores.attr, dir_name, parents_attr)
            if self.mask:
                # import ipdb; ipdb.set_trace()
                partial_scores.attr_mask = compute_f1(partial_scores.attr_mask,
                                                      parents_attr)
                plot_PR_from_munch(partial_scores.attr_mask,
                                   dir_name,
                                   parents_attr,
                                   mask=True)
        else:
            partial_scores = extract.produce_fake()
        # partial_scores = extract.produce_fake()

        if self.dist.rank() == 0:
            _text = 'Calculating FID between {}/{} [{}]... It may take a while'
            _text = _text.format(fid_calc[0], fid_calc[1], guide)
            progress_bar = tqdm(range(num_labels),
                                unit_scale=True,
                                total=num_labels,
                                desc=_text,
                                leave=self.verbose,
                                ncols=5)
            fid_results = {}
            prdc_results = {
                'perceptual_precision': {},
                'perceptual_recall': {},
                'perceptual_density': {},
                'perceptual_coverage': {}
            }

            for i in progress_bar:
                path = [folders[fid_calc[0]][i], folders[fid_calc[1]][i]]
                fid_results[attrs[i]] = FID(path,
                                            self.CUDA,
                                            domain=extract.attr[i],
                                            verbose=self.verbose)
                _prdc = prdc(path, verbose=self.verbose)
                for key in prdc_results.keys():
                    prdc_results[key][attrs[i]] = _prdc[key.replace(
                        'perceptual_', '')]
            if self.verbose:
                self.PRINT("Elapsed time: " + elapsed_time(Time))
        self.dist.barrier()
        partial_scores['fid'] = fid_results
        partial_scores['prdc'] = prdc_results
        return partial_scores
