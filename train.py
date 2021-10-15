from solver import Solver
import torch
import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
from termcolor import colored
from misc.utils import get_loss_value, unformated_text, save_json
from misc.utils import TimeNow, to_cuda, elapsed_time, isTimeWork, handle_error
from misc.losses import _GAN_LOSS
import torch.utils.data.distributed
import torch.nn.parallel
from torch import nn
import torch.nn.functional as F
from misc.scores import Scores
from PIL import Image
# import random
import re
from misc.mask_utils import cross_entropy2d
from misc.utils import get_batch_debug, get_screen_name, mean_std_tensor, denorm, save_img
from munch import Munch
from misc.visualization import debug_image_multidomain
warnings.filterwarnings('ignore')


class Train(Solver):
    def __init__(self, args, data_loader):
        super(Train, self).__init__(args, data_loader)
        if self.args.FAN:
            FAN = self.nets_ema.FAN
        else:
            FAN = None
        if self.dist.rank() == 0:
            self.scores = Scores(args,
                                 generator=self.nets_ema.G,
                                 style_model=self.nets_ema.S,
                                 mapping=self.nets_ema.F,
                                 verbose=self.verbose,
                                 FAN=FAN,
                                 mode='val')
        if self.args.TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.args.log_path)
        if self.args.VISDOM:
            from misc.utils import VisdomLinePlotter
            log_vis = get_screen_name(self.args.GPU[0])
            log_path = os.path.relpath(self.args.log_path)
            self.writer = VisdomLinePlotter(env_name=log_path, add_str=log_vis)
        self.lambda_ds = self.args.lambda_ds
        self.sample_epoch = self.args.sample_epoch
        self.run()
        if self.args.TENSORBOARD or self.args.VISDOM:
            self.writer.close()

        if not self.args.DISTRIBUTED:
            self.args.batch_size *= len(self.args.GPU)

    # ============================================================#
    # ============================================================#
    def update_loss(self, loss, value):
        try:
            self.LOSS[loss].append(value)
        except BaseException:
            self.LOSS[loss] = []
            self.LOSS[loss].append(value)

    # ============================================================#
    # ============================================================#
    def GAN_LOSS(self,
                 real_x,
                 fake_x,
                 label,
                 isFake=False,
                 fake_label=None,
                 model=None,
                 **kwargs):
        if model is None:
            model = self.nets.D
        _loss = _GAN_LOSS
        return _loss(model,
                     real_x,
                     fake_x,
                     label,
                     self.args,
                     isFake=isFake,
                     fake_label=fake_label,
                     **kwargs)

    # ============================================================#
    # ============================================================#
    def INFO(self, epoch, iter):
        # PRINT log info
        if self.verbose:
            if (iter + 1) % self.args.log_step == 0 or iter + epoch == 0:
                self.loss = {
                    key: get_loss_value(value)
                    for key, value in self.loss.items()
                }
                # for k in self.loss.keys():
                #     if 'Gsty' in k:
                #         color(self.loss, k, 'yellow')
                self.progress_bar.set_postfix(**self.loss)
            if (iter + 1) == len(self.data_loader):
                self.progress_bar.set_postfix('')

    # ============================================================#
    # ============================================================#
    def Decay_lr(self, current_epoch=0):
        self.lr -= (self.args.lr /
                    float(self.args.num_epochs - self.args.num_epochs_decay))
        self.f_lr -= (self.args.f_lr /
                      float(self.args.num_epochs - self.args.num_epochs_decay))
        self.update_lr(self.lr, self.f_lr)
        if self.verbose:  # and current_epoch % self.args.save_epoch == 0:
            _str = 'Decay lr to lr: {}, f_lr: {}.'
            self.PRINT(_str.format(self.lr, self.f_lr))

    # ============================================================#
    # ============================================================#
    def _compute_perceptual_loss(self, data_x, data_y):
        from misc.losses import _perceptual_loss
        return _perceptual_loss(self.vgg, data_x, data_y, self.args)

    # ============================================================#
    # ============================================================#
    def resample_policy(self, epoch, iter_per_epoch):
        if (self.args.ORG_DS or self.args.MASK_DIVERSITY
                or self.args.ATTR_DIVERSITY) and iter_per_epoch > 0:
            total_iter = 200000
            if iter_per_epoch * (epoch + 1) > total_iter:
                self.lambda_ds = 0.0
            else:
                org_lambda = self.args.lambda_ds
                num_epochs = (total_iter // iter_per_epoch) + 1
                self.lambda_ds = np.linspace(org_lambda, 0, num_epochs)[epoch]

        if isTimeWork() or epoch < 5:
            self.sample_epoch = self.args.sample_epoch
        else:
            if self.args.image_size == 256:
                self.sample_epoch = self.args.sample_epoch * 2

    # ============================================================#
    # ============================================================#
    def RESUME_INFO(self):
        image_path = self.args.sample_path
        model_path = self.args.model_save_path
        log_path = self.args.log_path
        log_file = colored(os.path.realpath(self.args.log), 'white')
        images_path = colored(os.path.realpath(image_path), 'white')
        model_path = colored(os.path.realpath(model_path), 'white')
        if self.args.TENSORBOARD:
            tensorboard_dir = os.path.realpath(log_path)
            tensorboard_command = 'tensorboard --logdir ' + tensorboard_dir
            log_command = colored(tensorboard_command, 'white')
            debug = 'Tensorboard'
        if self.args.VISDOM:
            visdom_dir = os.path.realpath(log_path)
            visdom_dir = '{}\n'.format(self.writer.env)
            visdom_dir += 'Please make sure of {} before running the code.'
            visdom_command = colored('python -m visdom.server', 'white')
            log_command = visdom_dir.format(visdom_command)
            debug = 'Visdom env_name'
        main_file = colored('python ' + ' '.join(sys.argv), 'white')
        self.PRINT("Main command: " + main_file)
        self.PRINT("Current time: " + colored(TimeNow(), 'white'))
        self.PRINT("Log txt: " + log_file)
        self.PRINT("Images saved at: " + images_path)
        self.PRINT("Model saved at: " + model_path)
        if self.args.TENSORBOARD or self.args.VISDOM:
            self.PRINT("{}: {}".format(debug, log_command))

        ll = get_batch_debug(self.data_loader_val, self.args.batch_size,
                             self.nets.G.random_noise)
        fixed_x, fixed_label, fixed_style, fixed_mask = ll

        self.inputs_val = Munch(x_src=fixed_x,
                                y_src=fixed_label,
                                f_src=fixed_style,
                                m_src=fixed_mask)
        if not self.args.pretrained_model:
            self.MISC(0, 0)
            return 0, 0
        pretrained_epoch = int(self.args.pretrained_model.split('_')[0])
        pretrained_iter = int(self.args.pretrained_model.split('_')[1])
        start = pretrained_epoch + 1
        total_iter = start * pretrained_iter
        for e in range(start):
            if e > self.args.num_epochs_decay:
                self.Decay_lr(e)
        return start, total_iter

    # ============================================================#
    # ============================================================#
    def validation_score(self, name, epoch, first=False):
        return
        not_validation = self.args.TRAIN_MASK
        not_validation = not_validation or self.args.ATTR == ''
        if not_validation:
            # It takes too much time to compute during training
            return
        for _str in ['Reference', 'Latent']:
            self.PRINT('--')
            fid, lpips, prdc = self.scores.Eval(
                name=name,
                latent_guided=_str == 'Latent',
                image_guided=_str == 'Reference',
                first=first)
            log = self.print_metric(epoch, fid, _str=_str, metric='FID')
            self.PRINT(log)
            log = self.print_metric(epoch, lpips, _str=_str, metric='LPIPS')
            self.PRINT(log)
            for key, values in prdc.items():
                log = self.print_metric(epoch,
                                        values,
                                        _str=_str,
                                        metric=key.upper())
                self.PRINT(log)

    # ============================================================#
    # ============================================================#
    def print_metric(self,
                     epoch,
                     dict_metric,
                     _str='',
                     metric='FID',
                     mode='VAL'):
        assert _str in ['Latent', 'Reference']
        _epoch = 'epoch_' + str(epoch).zfill(2)
        _metric = {}
        metric_json = {_epoch: {}}
        for key, value in dict_metric.items():
            if isinstance(value, dict):
                metric_json[_epoch][key] = {}
                for kk, vv in value.items():
                    metric_json[_epoch][key][kk] = {}
                    vv, std = mean_std_tensor(vv)
                    metric_json[_epoch][key][kk]['mean'] = vv
                    metric_json[_epoch][key][kk]['std'] = std
                    _metric['{}_{}'.format(key, kk)] = '{:.3f}'.format(vv)
                    self.LOSS['{}_{}_{}_{}'.format(metric, _str, key, kk)] = vv
            else:
                metric_json[_epoch][key] = {}
                value, std = mean_std_tensor(value)
                metric_json[_epoch][key]['mean'] = value
                metric_json[_epoch][key]['std'] = std
                _metric[key] = '{:.3f}'.format(value)
                self.LOSS['{}_{}_{}'.format(metric, _str, key)] = value
        log = "{0} - {2} - {1}\n -> {2} <-".format(metric, mode, '{}')
        log = log.format(_str,
                         ", ".join(": ".join(kv) for kv in _metric.items()))
        log = colored(log, 'yellow')
        json_file = self.args.json_file[metric]
        json_file = json_file.replace('.json', '_{}.json'.format(_str.lower()))
        save_json(metric_json, json_file)
        return log

    # ============================================================#
    # ============================================================#
    @torch.no_grad()
    def MISC(self, epoch, _iter):
        _start = epoch == 0 and _iter == 0
        _verbose = self.verbose
        _epoch = epoch + 1
        # save_model = (_epoch % self.args.model_epoch) == 0
        save_samples = (_epoch % self.sample_epoch) == 0
        eval_samples = (_epoch % self.args.eval_epoch) == 0
        save_tensorboard = (_epoch % self.args.log_epoch) == 0
        plot_debug = self.args.VISDOM or self.args.TENSORBOARD
        save_tensorboard = save_tensorboard and plot_debug
        if _verbose and not _start:
            # Save Weights
            self.save(epoch, _iter)

        if _verbose and save_samples and not _start:
            # Debug INFO
            elapsed = colored(elapsed_time(self.start_time), 'white')
            _str = '-> {} | Elapsed [Iter: {}] ({}/{}): {} | {}\n'
            self.Log = self.PRINT_LOG(self.args.batch_size, _print=False)
            log = _str.format(colored(TimeNow(), 'white'), self.total_iter,
                              colored(str(epoch), 'white'),
                              self.args.num_epochs, elapsed, self.Log)

            loss_log = []
            for tag, value in sorted(self.LOSS.items()):
                loss_log.append("{}: {:.4f}".format(tag,
                                                    np.array(value).mean()))
            loss_log = ', '.join(loss_log)
            loss_log = colored(loss_log, 'blue')
            log += loss_log
            self.PRINT(log)

        if _verbose and (save_samples or _start):
            if _start and self.args.FAN:
                # import ipdb; ipdb.set_trace()
                data = iter(self.data_loader_val).next()
                x_src = to_cuda(data['image'])
                m_src = to_cuda(data['mask'])
                if self.args.dataset == 'DeepFashion2':
                    if not self.args.TRAIN_MASK:
                        org_x = denorm(x_src.clone())
                        x0 = kp_src.clone()
                        x0 = (x0 * org_x).expand_as(org_x)
                        list_h = torch.cat([org_x, x0], dim=3)
                        file_h = os.path.join(self.args.sample_path,
                                              'keypoints.jpg')
                        print(f'Saving Keypoint visualization on: {file_h}.')
                        save_img(list_h, 3, file_h, denormalize=False)
                else:
                    if self.args.TRAIN_MASK:
                        img_heatmap = m_src
                        self.nets.FAN.get_visualization(
                            to_cuda(img_heatmap),
                            current=True,
                            dirname=self.args.sample_path,
                            mask=x_src)
                    else:
                        img_heatmap = x_src
                    self.nets.FAN.get_visualization(
                        to_cuda(img_heatmap),
                        current=True,
                        dirname=self.args.sample_path)
            name = self.output_sample(epoch, _iter)
            try:
                img_debugs = debug_image_multidomain(self.nets_ema, self.args,
                                                     self.data_loader_val,
                                                     name)
            except Exception as e:
                print(str(e))
                raise e
                # pass

        # if _verbose and (eval_samples or _start):
        if _verbose and eval_samples:
            _name = self.output_sample(epoch, _iter).split('.jpg')[0]
            self.validation_score(_name, epoch, first=_start)

        if _verbose and save_tensorboard:
            if 'log' in locals():
                self.writer.text('Last_epoch', unformated_text(log))
            for tag, value in sorted(self.LOSS.items()):
                self.writer.add_scalar('loss/' + tag,
                                       np.array(value).mean(),
                                       global_step=epoch)
            if 'img_debugs' in locals():
                for mode, img_files_mode in enumerate(img_debugs):
                    if img_files_mode is None:
                        continue
                    big_size = self.args.image_size == 512
                    big_size = big_size and self.args.mode == 'train'
                    if big_size and self.args.VISDOM:
                        imgs = [
                            np.array(
                                Image.open(i).convert('RGB').resize(
                                    (Image.open(i).size[0] // 2,
                                     Image.open(i).size[1] // 2),
                                    Image.LANCZOS)) for i in img_files_mode
                        ]
                    else:
                        imgs = [
                            np.array(Image.open(i).convert('RGB'))
                            for i in img_files_mode
                        ]
                    title = list(
                        set([
                            re.sub(r'_style_\w.jpg', '', os.path.basename(i))
                            for i in img_files_mode
                        ]))
                    assert len(title) == 1
                    title = title[0]
                    imgs = [np.expand_dims(i, axis=0) for i in imgs]
                    imgs = [np.transpose(i, (0, 3, 1, 2)) for i in imgs]
                    imgs = np.concatenate(imgs, axis=0) / 255.
                    self.writer.add_images('last_epoch_mode_{}'.format(mode),
                                           imgs,
                                           0,
                                           title=title,
                                           caption=title)
        self.dist.barrier()
        self.resample_policy(epoch, _iter)
        # Decay learning rate
        if epoch > self.args.num_epochs_decay:
            self.Decay_lr(epoch)

    # ============================================================#
    # ============================================================#
    def reset_losses(self):
        return {}

    # ============================================================#
    # ============================================================#
    def current_losses(self, mode, latent=False, guided=None, **kwargs):
        loss = 0
        if latent:
            _str = '_l'
        elif guided is not None:
            _str = '_g'
        for key, _ in kwargs.items():
            if mode in key and _str in key:
                loss += self.loss[key]
                self.update_loss(key, get_loss_value(self.loss[key]))
        return loss

    # ============================================================#
    # ============================================================#
    def _update_ema_model(self, model, ema_model):
        ema_beta = 0.999  # self.args.ema_beta
        model_params = dict(model.named_parameters())
        ema_model_params = dict(ema_model.named_parameters())
        for key in model_params:
            ema_model_params[key].data.mul_(ema_beta).add_(
                1 - ema_beta, model_params[key].data)

    # ============================================================#
    # ============================================================#
    def _update_ema_models(self):
        if self.dist.rank() == 0:
            self._update_ema_model(self.nets.G, self.nets_ema.G)
            self._update_ema_model(self.nets.S, self.nets_ema.S)
            self._update_ema_model(self.nets.F, self.nets_ema.F)

    # ============================================================#
    # ============================================================#
    def to_cuda(self, *args):
        vars = []
        for arg in args:
            vars.append(to_cuda(arg))
        return vars

    # ============================================================#
    # ============================================================#
    def name_format(self, _str, pyramid_id=None, latent=False, guided=None):
        if latent:
            _str += '_l'
        elif guided is not None:
            _str += '_g'
        if pyramid_id is None:
            return _str
        else:
            return _str + '_' + str(pyramid_id)

    # ============================================================#
    # ============================================================#
    def train_model(
            self,
            generator=False,
            style=False,
            mapping=False,
            discriminator=False,
    ):

        for p in self.nets.D.parameters():
            p.requires_grad_(discriminator)

        for p in self.nets.G.parameters():
            p.requires_grad_(generator)

        for p in self.nets.F.parameters():
            p.requires_grad_(mapping)

        for p in self.nets.S.parameters():
            p.requires_grad_(style)

    # ============================================================#
    # ============================================================#
    def Dis_update(self,
                   real_x,
                   real_c,
                   fake_c,
                   fan=None,
                   mask=None,
                   latent=False,
                   guided=None,
                   guided_mask=None):
        self.train_model(discriminator=True)
        self.reset_grad()

        # Train Variables
        # ======================================================#
        fake_rgb = None

        g_input = real_x

        # ======================================================#
        with torch.no_grad():
            # Latent or Guided forward
            if latent:
                z_fake = to_cuda(self.nets.G.random_noise(real_x))
                style_fake = self.nets.F(z_fake, fake_c)
                # import ipdb; ipdb.set_trace()
            elif guided is not None:
                style_fake = self.nets.S(guided, fake_c)

            # ======================================================#
            # Fake image
            content_real = self.nets.G.encoder(g_input, fan=fan)
            fake_x = self.nets.G.decoder(content_real, style_fake, fan=fan)

            if self.args.TRAIN_MASK:
                fake_x = nn.Softmax(dim=1)(fake_x)

        # Adversarial Loss
        gan_loss = self.GAN_LOSS(
            real_x,
            fake_x,
            real_c,
            fake_label=fake_c,
            seg=mask,
            fake_seg=fake_rgb,
        )

        name_src = self.name_format('Dsrc', latent=latent, guided=guided)
        self.loss[name_src] = gan_loss['src'] * self.args.lambda_src
        if 'dr1' in gan_loss.keys():
            name_d1 = self.name_format('Dr1', latent=latent, guided=guided)
            self.loss[name_d1] = gan_loss['dr1']

        d_loss = self.current_losses('D',
                                     latent=latent,
                                     guided=guided,
                                     **self.loss)
        d_loss.backward()
        self.optims['D'].step()
        self.GPU_MEMORY_USED = self.get_gpu_memory_used()

    # ============================================================#
    # ============================================================#
    def Gen_update(self,
                   real_x,
                   real_c,
                   fake_c,
                   fan=None,
                   mask=None,
                   latent=False,
                   guided=None,
                   guided_mask=None,
                   guided_c=None):
        self.reset_grad()
        criterion_l1 = nn.L1Loss()
        DIV = self.args.DS or self.args.ORG_DS or self.args.MASK_DIVERSITY
        DIV = DIV or self.args.ATTR_DIVERSITY
        # Train Variables
        # ======================================================#
        fake_rgb = None
        real_mask = mask

        # ======================================================#
        # Latent or Guided forward
        if latent:
            z_fake = to_cuda(self.nets.G.random_noise(real_x))
            style_fake = self.nets.F(z_fake, fake_c)
        elif guided is not None:
            style_fake = self.nets.S(guided, fake_c)

        style_org = self.nets.S(real_x, real_c)
        if not self.args.STARGAN_TRAINING:
            style_org = style_org.detach()

        # ======================================================#
        g_input = real_x

        # Fake image
        content_real = self.nets.G.encoder(g_input, fan=fan)
        fake_x = self.nets.G.decoder(content_real, style_fake, fan=fan)

        if self.args.TRAIN_MASK:
            fake_x = nn.Softmax(dim=1)(fake_x)

        g_input_rec = fake_x

        # Adversarial Loss
        gan_loss = self.GAN_LOSS(
            fake_x,
            real_x,
            fake_c,
            isFake=True,
            seg=fake_rgb,
            fake_label=real_c,
            fake_seg=real_mask,
            feat_loss=self.args.FeatLoss and guided is not None,
        )

        name_src = self.name_format('Gsrc', latent=latent, guided=guided)
        self.loss[name_src] = gan_loss['src'] * self.args.lambda_src

        # Feat Loss
        if 'feat' in gan_loss.keys():
            name_ft = self.name_format('Gft', latent=latent, guided=guided)
            self.loss[name_ft] = gan_loss['feat'] * self.args.lambda_feat

        # Diversity Loss
        # import ipdb; ipdb.set_trace()
        if DIV and self.lambda_ds > 0.0:
            # TODO for rgb_semantics
            if latent:
                z_fake2 = to_cuda(self.nets.G.random_noise(real_x))
                style_fake2 = self.nets.F(z_fake2, fake_c)
            elif guided is not None:
                randperm = torch.randperm(guided.size(0))
                style_fake2 = self.nets.S(guided[randperm], fake_c[randperm])
            fake_x2 = self.nets.G.decoder(content_real, style_fake2, fan=fan)
            if self.args.TRAIN_MASK:
                fake_x2 = nn.Softmax(dim=1)(fake_x2)

            if self.args.ORG_DS:
                # StarGAN2 DS loss
                name_sd = self.name_format('Gsd', latent=latent, guided=guided)
                if (self.args.REENACTMENT
                        or self.args.TRAIN_MASK) and self.args.FAN:
                    _fan = 1 - F.interpolate(
                        fan[0], size=fake_x.size(2), mode='bilinear')
                    _fan = _fan.expand_as(fake_x)
                    _fake_x = _fan * fake_x
                    _fake_x2 = _fan * fake_x2
                    g_loss_ds = -criterion_l1(_fake_x[_fan > 0],
                                              _fake_x2[_fan > 0].detach())
                else:
                    g_loss_ds = -criterion_l1(fake_x, fake_x2.detach())
                self.loss[name_sd] = g_loss_ds * self.lambda_ds

        # Style Reconstruction
        style_fake_rec = self.nets.S(fake_x, fake_c)

        if self.args.SPLIT_STYLE:
            g_loss_sty = criterion_l1(style_fake[:, 0], style_fake_rec[:, 0])
            for i in range(1, style_fake.size(1)):
                g_loss_sty += criterion_l1(
                    style_fake[:, i, :self.args.small_dim],
                    style_fake_rec[:, i, :self.args.small_dim])
            # g_loss_sty = criterion_l1(style_fake, style_fake_rec)
        else:
            if self.args.dataset != 'CelebA_HQ':
                g_loss_sty = criterion_l1(style_fake[fake_c == 1],
                                          style_fake_rec[fake_c == 1])
            else:
                g_loss_sty = criterion_l1(style_fake, style_fake_rec)

        name_sty = self.name_format('Gsty', latent=latent, guided=guided)
        self.loss[name_sty] = self.args.lambda_sty * g_loss_sty

        # Cycle-Consistency Loss
        # fake_fan = self.nets.FAN.get_heatmap(fake_x)
        rec_x = self.nets.G(g_input_rec, style_org, fan=fan)
        if self.args.TRAIN_MASK:
            g_loss_rec = cross_entropy2d(rec_x, real_x)
        else:
            g_loss_rec = criterion_l1(rec_x, real_x)
        loss_rec = self.args.lambda_rec * g_loss_rec
        name_rec = self.name_format('Grec', latent=latent, guided=guided)
        self.loss[name_rec] = loss_rec

        if not self.args.TRAIN_MASK and self.args.REENACTMENT:
            _fan = F.interpolate(fan[0], size=fake_x.size(2), mode='bilinear')
            _fan = _fan.expand_as(fake_x)
            _fake_x = _fan * fake_x
            _real_x = _fan * real_x
            g_loss_col = criterion_l1(_fake_x[_fan > 0], _real_x[_fan > 0])
            loss_rec = self.args.lambda_color * g_loss_col
            name_rec = self.name_format('Gcol', latent=latent, guided=guided)
            self.loss[name_rec] = loss_rec

        g_loss = self.current_losses('G',
                                     latent=latent,
                                     guided=guided,
                                     **self.loss)
        g_loss.backward()
        self.GPU_MEMORY_USED = self.get_gpu_memory_used()

    # ============================================================#
    # ============================================================#
    @handle_error
    def run(self):
        # lr cache for decaying
        self.lr = self.args.lr
        self.f_lr = self.args.f_lr

        _string = 'Training with lr: {}, f_lr: {}.'
        self.PRINT(
            _string.format(self.optims.G.param_groups[0]['lr'],
                           self.optims.F.param_groups[0]['lr']))

        # Start with trained info if exists
        self.LOSS = {}
        self.start_time = time.time()
        start, self.total_iter = self.RESUME_INFO()
        self.resample_policy(start, self.total_iter)

        # _name = self.output_sample(start, 0).split('.jpg')[0]
        # self.validation_score(_name, start, first=True)

        # Log info
        self.Log = self.PRINT_LOG(self.args.batch_size)

        # Start training
        self.flag_stop = False
        for epoch in range(start, self.args.num_epochs):
            self.LOSS = {}
            desc_bar = '[Iter: %08d] Epoch: %02d/%02d' % (
                self.total_iter, epoch, self.args.num_epochs)
            self.progress_bar = tqdm(
                enumerate(self.data_loader),
                unit_scale=True,
                total=len(self.data_loader),
                desc=desc_bar,
                disable=not self.verbose,
                leave=not ((epoch + 1) % self.sample_epoch),
                ncols=10)
            # for _iter, (real_x, real_c, mask, keyp) in self.progress_bar:
            for _iter, data in self.progress_bar:
                real_x, real_c = data['image'], data['label']
                real_x_ref, real_c_ref = data['image_ref'], data['label_ref']
                mask, mask_ref = data['mask'], data['mask_ref']
                self.loss = self.reset_losses()
                self.total_iter += 1 * self.dist.size()

                shuffle_order = torch.randperm(real_c.size(0))
                real_x = real_x[shuffle_order]
                real_c = real_c[shuffle_order]
                mask = mask[shuffle_order]
                real_x_ref = real_x_ref[shuffle_order]
                real_c_ref = real_c_ref[shuffle_order]
                mask_ref = mask_ref[shuffle_order]

                if (real_c == 0).all():
                    continue
                if self.args.ATTR and (real_c[:, 0] == 0).all():
                    continue

                # Cuda
                # ============================================================#
                real_x, real_c = self.to_cuda(real_x, real_c)
                real_x_ref, real_c_ref = self.to_cuda(real_x_ref, real_c_ref)
                mask = to_cuda(mask)
                mask_ref = to_cuda(mask_ref)

                # FAN
                # ============================================================#
                if self.args.FAN:
                    if self.args.TRAIN_MASK:
                        img_heatmap = mask  # it is real_x with different name
                    else:
                        img_heatmap = real_x
                    fan = self.nets.FAN.get_heatmap(img_heatmap)
                    fan0 = fan1 = fan
                else:
                    fan0 = fan1 = None

                real_x0 = real_x1 = real_x
                real_c0 = real_c1 = real_c
                mask0 = mask1 = mask
                _invert = torch.arange(real_c.size(0) - 1, -1, -1)
                guided_x0 = guided_x1 = real_x_ref  # real_x[_invert]
                guided_c0 = guided_c1 = real_c_ref  # real_c[_invert]
                guided_mask0 = guided_mask1 = mask_ref  # mask[_invert]
                # ============================================================#
                # ======================== Train D ===========================#
                # ============================================================#
                self.Dis_update(real_x0,
                                real_c0,
                                guided_c0,
                                fan=fan0,
                                mask=mask0,
                                latent=True)
                if self.args.STARGAN_TRAINING:
                    self.Dis_update(real_x0,
                                    real_c0,
                                    guided_c0,
                                    fan=fan0,
                                    mask=mask0,
                                    guided=guided_x0,
                                    guided_mask=guided_mask0)

                if self.dist.rank() == 0:
                    self._update_ema_model(self.nets.D, self.nets_ema.D)

                # ============================================================#
                # ======================== Train G ===========================#
                # ============================================================#
                self.train_model(generator=True, mapping=True, style=True)
                self.Gen_update(real_x1,
                                real_c1,
                                guided_c1,
                                fan=fan1,
                                mask=mask1,
                                latent=True)
                self.optims['G'].step()
                self.optims['F'].step()
                self.optims['S'].step()

                if self.args.STARGAN_TRAINING:
                    self.train_model(generator=True)
                    self.Gen_update(real_x1,
                                    real_c1,
                                    guided_c1,
                                    fan=fan1,
                                    mask=mask1,
                                    guided=guided_x1,
                                    guided_mask=guided_mask1)
                    self.optims['G'].step()
                self._update_ema_models()
                # ====================== DEBUG =====================#
                self.GPU_MEMORY_USED = self.get_gpu_memory_used()
                self.INFO(epoch, _iter)
                # self.MISC(epoch, _iter)

                if (_iter % self.args.sample_iter
                    ) == 0 and _iter > 0 and self.args.sample_iter > 0:
                    name = self.output_sample(epoch, _iter)
                    debug_image_multidomain(self.nets_ema, self.args,
                                            self.data_loader_val, name)

            # ============================================================#
            # ======================= MISCELANEOUS =======================#
            # ============================================================#
            # Shuffling dataset each epoch
            self.data_loader.dataset.shuffle(epoch)
            self.MISC(epoch, _iter)


# Ablation studies qualitative results
# bs: snapshot/samples/CelebA_HQ/ORG_DS/FAN/STARGAN_TRAINING/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS
# A:  snapshot/samples/CelebA_HQ/ORG_DS/TRAIN_MASK/STYLE_SEMANTICS/FAN/STARGAN_TRAINING/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS/lambda_ds_20.0
# B:  snapshot/samples/CelebA_HQ/ORG_DS/TRAIN_MASK/STYLE_SEMANTICS/FAN/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS/lambda_ds_20.0
# C:  snapshot/samples/CelebA_HQ/ORG_DS/MOD/TRAIN_MASK/STYLE_SEMANTICS/FAN/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS/lambda_ds_20.0
# D:  snapshot/samples/CelebA_HQ/ORG_DS/MOD/TRAIN_MASK/STYLE_SEMANTICS/FAN/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS/SPLIT_STYLE/lambda_ds_20.0
# D+:
# snapshot/samples/CelebA_HQ/ORG_DS/MOD/FAN/STARGAN_TRAINING/HAIR/GENDER/EYEGLASSES/EARRINGS/HAT/BANGS/SPLIT_STYLE
