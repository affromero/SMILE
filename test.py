from solver import Solver
import warnings
from misc.utils import mean_std_tensor
from misc.scores import Scores
from termcolor import colored
import os
from misc.utils import TimeNow_str
from data_loader import get_loader
from misc.visualization import debug_image_multidomain
from misc.utils import save_json
import math
warnings.filterwarnings('ignore')


class Test(Solver):
    def __init__(self, args, data_loader):
        self.args = args
        super().__init__(args, data_loader)

    # ==================================================================#
    # ==================================================================#
    def sample(self, dataset='', load=False):
        last_name = self.resume_name()
        save_folder = os.path.join(self.args.sample_path,
                                   '{}_test'.format(last_name))
        os.makedirs(save_folder, exist_ok=True)
        # max(16, self.args.batch_size)
        batch_size = self.args.batch_sample
        data_loader = get_loader(self.args,
                                 batch_size=batch_size,
                                 shuffling=True)

        string = TimeNow_str()
        name = os.path.join(save_folder, string)
        self.PRINT(
            'Translated test images and saved into "{}"..!'.format(name))

        if not self.args.REENACTMENT:
            debug_image_multidomain(self.nets_ema,
                                    self.args,
                                    data_loader,
                                    name,
                                    training=False,
                                    translate_all=True,
                                    fill_rgb=self.args.FILL_RGB)

        debug_image_multidomain(self.nets_ema,
                                self.args,
                                data_loader,
                                name,
                                training=False,
                                fill_rgb=self.args.FILL_RGB)

        # if not self.args.REENACTMENT:
        #     debug_image_multidomain(self.nets_ema,
        #                             self.args,
        #                             data_loader,
        #                             name,
        #                             training=False,
        #                             translate_all=True,
        #                             fill_rgb=self.args.FILL_RGB)

    def print_metric(self, dict_metric, _str='', metric='FID', mode='TEST'):
        assert _str in ['Latent', 'Reference']
        _metric = {}
        for key, value in dict_metric.items():
            _metric[key] = {}
            if isinstance(value, dict):
                for kk, vv in value.items():
                    vv, std = mean_std_tensor(vv)
                    _metric[key][kk] = {}
                    _metric[key][kk]['mean'] = '{:.3f}'.format(
                        vv) if not isinstance(vv, str) else vv
                    _metric[key][kk]['std'] = '{:.3f}'.format(std)
            else:
                value, std = mean_std_tensor(value)
                _metric[key]['mean'] = '{:.3f}'.format(value)
                _metric[key]['std'] = '{:.3f}'.format(std)
                # _metric[key] = '{:.3f}'.format(value)
        log = "{0} - {2} - {1}\n ->\n{2}\n<-".format(metric, mode, '{}')
        log = log.format(
            _str,
            "\n".join("\t{}: {}".format(k, v) for k, v in _metric.items()))
        log = colored(log, 'yellow')
        return log, _metric

    def Eval(self):
        if self.args.FAN:
            FAN = self.nets_ema.FAN
        else:
            FAN = None
        scores = Scores(self.args,
                        generator=self.nets_ema.G,
                        style_model=self.nets_ema.S,
                        mapping=self.nets_ema.F,
                        verbose=True,
                        FAN=FAN,
                        mode='test')
        results_json = {}

        # for _str in ['Reference', 'Latent']:
        for _str in ['Latent', 'Reference']:
            results_json[_str] = {}
            results = scores.Eval(latent_guided=_str == 'Latent',
                                  image_guided=_str == 'Reference')
            for keys in results.keys():

                if 'files' in keys:
                    continue
                if 'Female' in results[keys].keys():
                    log, _metric = self.print_metric(results[keys],
                                                     _str=_str,
                                                     metric=keys.upper())
                    results_json[_str][keys.upper()] = _metric
                    scores.PRINT(log)
                else:
                    for key, values in results[keys].items():
                        if key in ['P', 'R']:
                            continue  # not interested in mean/sd of precision and recall

                        _name = '{}_{}'.format(keys.upper(), key.upper())
                        log, _metric = self.print_metric(values,
                                                         _str=_str,
                                                         metric=_name)
                        results_json[_str][_name] = _metric
                        scores.PRINT(log)
        json_file = '{}/{}_test_{}'.format(self.args.sample_path,
                                           self.args.pretrained_model,
                                           self.args.json_file)
        save_json(results_json, json_file)


# python main.py --batch_size=4 --GPU=NO_CUDA --FAN --EYEGLASSES --GENDER
# --HAT --EARRINGS --HAIR --BANGS --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS
# --lambda_ds=20 --MOD --SPLIT_STYLE
