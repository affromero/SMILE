# ==================================================================#
# ==================================================================#
import torch
import os
from tqdm import tqdm
from misc.utils import to_cuda
from misc.ops import ops
import random
from metrics.lpips import LPIPS  # calculate_lpips_given_images
from metrics.pose_model import Hopenet  # calculate_pose_given_images
from metrics.attribute_model import AttNet  # calculate_attr_given_images
from metrics.smileSYN import SMILE_SYN as SMILE
from metrics.f1_score import compute_f1, plot_PR
from misc.mask_utils import label2mask, label2mask_plain, scatterMask
from metrics.segmentation_model import MaskNet
from metrics.iou import IoU, get_mean_for_iou
from munch import Munch
from misc.utils import denorm
import pickle


def save_image(folder, name, image, real_loc='', denormalize=True):
    from misc.utils import denorm
    from PIL import Image
    import os
    import numpy as np
    new_file = os.path.join(folder, name)
    if real_loc:
        if os.path.islink(new_file) or os.path.isfile(new_file):
            os.remove(new_file)
        # if not os.path.islink(new_file):
        #     os.symlink(real_loc, new_file)
        os.symlink(os.path.abspath(real_loc), new_file)
        return
    if denormalize:
        image = denorm(image)

    img = image.cpu()[0].numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).resize((256, 256)).save(new_file)


def plot_PR_from_munch(data_munch, dir_name, parents_attr, mask=False):
    for attr in data_munch['f1'].keys():
        # import ipdb; ipdb.set_trace()
        value = Munch({k: data_munch[k][attr] for k in data_munch.keys()})
        plot_PR(value, dir_name, parents_attr, attr, mask=mask)


class Extract(ops):
    def __init__(self,
                 mode,
                 data_loader,
                 generator,
                 style_model,
                 mapping,
                 dir_name,
                 MODE='val',
                 data_loader_train=None,
                 n_fakes=10,
                 FAN=None,
                 mask=False,
                 verbose=False):
        self.data_loader = data_loader
        self.data_loader_train = data_loader_train
        self.dir_name = dir_name
        self.mode = mode
        assert self.mode in ['reference', 'latent']
        assert style_model is not None
        self.MODE = MODE
        self.generator = generator
        self.encoder = style_model
        self.mapping = mapping
        self.n_fakes = n_fakes
        self.verbose = verbose
        self.FAN = FAN
        self.mask = mask
        self.attr = self.data_loader.dataset.selected_attrs
        self.num_labels = len(self.attr)

    @torch.no_grad()
    def create_folders(self):
        self.folders = {}

        for ff in ['real', 'fake', 'guided', 'guided_match']:
            self.folders[ff] = [
                os.path.join(self.dir_name, '{}_{}'.format(ff, attr))
                for attr in self.attr
            ]

            if self.mask:
                self.folders[ff + '_mask'] = [
                    os.path.join(self.dir_name, '{}_mask_{}'.format(ff, attr))
                    for attr in self.attr
                ]

        if self.data_loader_train is not None:
            self.folders['train'] = [
                os.path.join(self.dir_name, 'train_{}'.format(attr))
                for attr in self.attr
            ]

            if self.mask:
                self.folders['train_mask'] = [
                    os.path.join(self.dir_name, 'train_mask_{}'.format(attr))
                    for attr in self.attr
                ]

        for ff in self.folders.keys():
            for f in self.folders[ff]:
                os.makedirs(f, exist_ok=True)

        return self.folders

    @torch.no_grad()
    def produce_fake(self):
        dataset = self.data_loader.dataset
        if self.data_loader_train is not None:
            # for i, (_real_x, _org_c, _, _files) in tqdm(
            for i, data in tqdm(enumerate(self.data_loader_train),
                                desc='Symlinks from training images',
                                total=len(self.data_loader_train),
                                ncols=5,
                                leave=self.verbose):
                _real_x, _org_c, = data['image'], data['label']
                _mask, _files = data['mask'], data['filename']
                for real_x, org_c, mask, files in zip(_real_x, _org_c, _mask,
                                                      _files):
                    real_x = real_x.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                    for idx, label in enumerate(org_c):
                        if label.item():
                            file_img = str(i + 1).zfill(5) + '.png'
                            folder_real = self.folders['train'][idx]
                            save_image(folder_real,
                                       file_img,
                                       real_x,
                                       real_loc=files)
                            if self.mask:
                                maskname = files.replace(
                                    dataset.data_dir, dataset.mask_dir)
                                maskname = maskname.replace('jpg', 'png')
                                folder_real = self.folders['train_mask'][idx]
                                save_image(folder_real,
                                           file_img,
                                           mask,
                                           real_loc=maskname)

        count_img = 0
        idx_seed = 0
        lpips = to_cuda(LPIPS(), fixed=True)
        posenet = to_cuda(Hopenet(), fixed=True)
        attnet = to_cuda(AttNet(), fixed=True)
        masknet = to_cuda(MaskNet(), fixed=True)
        if self.mask:
            attnet_mask = to_cuda(AttNet(mask=True), fixed=True)
            smile = to_cuda(SMILE(), fixed=True)
        lpips_score, attr_score = {}, {'real': {}, 'fake': {}}
        pose_score = {'yaw': {}, 'pitch': {}, 'roll': {}}
        iou_score = {}
        if self.mask:
            name_org = '_mask'
            name_mask = ''
            attr_mask_score = {'real': {}, 'fake': {}}
        else:
            name_mask = '_mask'
            name_org = ''
        # for (_real_x, _org_c, _mask, _files) in tqdm(
        all_files = {}
        for data in tqdm(self.data_loader,
                         desc='Creating fake images [{} - {}]'.format(
                             dataset.mode, self.mode),
                         total=len(self.data_loader),
                         ncols=5,
                         leave=self.verbose):
            _real_x, _org_c = data['image'], data['label']
            _mask, _files = data['mask'], data['filename']
            for real_x, org_c, mask, files in zip(_real_x, _org_c, _mask,
                                                  _files):
                count_img += 1
                real_x = to_cuda(real_x.unsqueeze(0))
                mask = to_cuda(mask.unsqueeze(0))
                # import ipdb; ipdb.set_trace()
                for idx, label in enumerate(org_c):
                    file_img = str(count_img).zfill(5) + '.png'
                    if label.item() == 1 and len(org_c) > 1:
                        folder_real = self.folders['real'][idx]
                        if os.path.isfile(
                                file_img) and self.dist.is_available():
                            continue
                        save_image(folder_real,
                                   file_img,
                                   real_x,
                                   real_loc=files)
                        if self.mask:
                            maskname = files.replace(dataset.data_dir,
                                                     dataset.mask_dir)
                            maskname = maskname.replace('jpg', 'png')
                            folder_real = self.folders['real_mask'][idx]
                            save_image(
                                folder_real,
                                file_img,
                                label2mask(real_x, n=real_x.size(1)),
                                # real_loc=maskname,
                                denormalize=False)
                        continue
                    target_c = to_cuda(org_c.clone().unsqueeze(0))
                    if self.data_loader.dataset.name in self.OneDomainDatasets(
                    ):
                        target_c = torch.zeros_like(target_c)
                    target_c[:, idx] = 1
                    target_c = self.target_multiAttr(target_c, index=idx)

                    label_guided = to_cuda(org_c.clone().unsqueeze(
                        0))  # it only matters the idx label
                    # import ipdb; ipdb.set_trace()
                    label_guided = 1 - label_guided
                    # label_guided[:, 1 + idx - idx%2] = 1 - label_guided[:, 1 - idx + idx%2]

                    content_real = self.generator.encoder(real_x)

                    style_org = self.encoder(real_x, org_c)
                    if self.FAN is not None:
                        fan_input = real_x if not self.mask else mask
                        fan = self.FAN.get_heatmap(fan_input)
                    else:
                        fan = None

                    fake_images = []
                    rec_images = []
                    if self.mask:
                        sem_images = []
                    for n in range(self.n_fakes):
                        file_img_fake = '{}_{}.png'.format(
                            str(count_img).zfill(5),
                            str(n).zfill(2))
                        if self.mode == 'latent':
                            noise = to_cuda(
                                self.generator.random_noise(real_x.size(0)))
                            style = self.mapping(noise, label_guided)
                        else:
                            # import ipdb; ipdb.set_trace()
                            not_attr_files = dataset.attr2filenames[
                                self.attr[idx]]
                            random.seed(idx_seed)
                            random_idx = random.randint(
                                0,
                                len(not_attr_files) - 1)
                            idx_seed += 1
                            file_guided = not_attr_files[random_idx]
                            file_guided = os.path.abspath(
                                os.path.join(dataset.data_dir, file_guided))
                            assert os.path.isfile(file_guided)
                            if self.mask:
                                img_guided = dataset.get_mask_from_file(
                                    file_guided).unsqueeze(0)
                            else:
                                img_guided = dataset.file2img(
                                    file_guided).unsqueeze(0)
                            folder_guided = self.folders['guided'][idx]
                            save_image(folder_guided,
                                       os.path.basename(file_guided),
                                       img_guided,
                                       real_loc=file_guided)
                            folder_guided = self.folders['guided_match'][idx]
                            save_image(folder_guided,
                                       file_img_fake,
                                       img_guided,
                                       real_loc=file_guided)
                            if self.mask:
                                rgb_guided = dataset.file2img(
                                    file_guided).unsqueeze(0)
                                rgb_guided = to_cuda(rgb_guided)
                                folder_guided = self.folders['guided_mask'][
                                    idx]
                                save_image(folder_guided,
                                           os.path.basename(file_guided),
                                           label2mask(img_guided,
                                                      n=img_guided.size(1)),
                                           denormalize=False
                                           # real_loc=maskname,
                                           )
                                folder_guided = self.folders[
                                    'guided_match_mask'][idx]
                                save_image(
                                    folder_guided,
                                    file_img_fake,
                                    label2mask(img_guided,
                                               n=img_guided.size(1)),
                                    # real_loc=maskname,
                                    denormalize=False)
                            img_guided = to_cuda(img_guided)
                            style = self.encoder(img_guided, label_guided)

                        style_fake = style_org.clone()
                        style_fake[:, idx // 2] = style[:, idx // 2]
                        fake_x = self.generator.decoder(content_real,
                                                        style_fake,
                                                        fan=fan)
                        folder_fake = self.folders['fake' + name_org][idx]
                        if self.mask:
                            _fake_x = label2mask(fake_x, n=fake_x.size(1))
                            save_image(folder_fake,
                                       file_img_fake,
                                       _fake_x,
                                       denormalize=False)
                            folder_fake = self.folders['fake' + name_mask][idx]
                            fake_x = scatterMask(
                                label2mask_plain(fake_x)[:, 0], fake_x.size(1))
                            sem_images.append(fake_x)
                            rec_x = self.generator(fake_x, style_org, fan=fan)
                            rec_x = scatterMask(
                                label2mask_plain(rec_x)[:, 0], rec_x.size(1))
                            rec_images.append(rec_x)
                            if self.mode == 'latent':
                                fake_smile = smile.forward_from_tensor(
                                    fake_x, style_random=True)
                            else:
                                current_attr = dataset.selected_attrs[idx]
                                attr2mask = ','.join(
                                    dataset.mask_attr[current_attr])
                                style_ref = smile.model.encoder(
                                    rgb_guided, img_guided)
                                fake_smile = smile.forward_from_tensor(
                                    fake_x,
                                    rgb_guide=mask,
                                    sem_guide=real_x,
                                    style_ref=style_ref,
                                    attr=attr2mask)
                            save_image(folder_fake,
                                       file_img_fake,
                                       fake_smile,
                                       denormalize=False)
                            # fake_smile = denorm(fake_smile)
                            fake_images.append(fake_smile)
                        else:
                            rec_x = self.generator(fake_x, style_org, fan=fan)
                            rec_images.append(rec_x)
                            save_image(folder_fake, file_img_fake, fake_x)
                            # fake_x = denorm(fake_x)
                            fake_images.append(fake_x)
                        # import ipdb; ipdb.set_trace()
                        # attr_score = pickle.load(open('dummy_attr.pkl', 'rb'))
                        # attr_score = compute_f1(attr_score)
                        # plot_PR(attr_score[self.attr[0]], self.dir_name, list(dataset.parent_attrs.keys()), self.attr[0])
                        # break
                    # import ipdb; ipdb.set_trace()
                    # break
                    lpips_value = lpips.calculate_lpips_given_images(
                        fake_images)  # denorm internally
                    pose_fake = posenet.calculate_pose_given_images(
                        fake_images)  # denorm internally
                    pose_real = posenet(real_x) if not self.mask else posenet(
                        mask)
                    pose_error = posenet.compute_loss(pose_real,
                                                      pose_fake,
                                                      output=True)
                    yaw_error, pitch_error, roll_error = pose_error.chunk(3)
                    attr_fake = attnet.calculate_attr_given_images(
                        fake_images)  # denorm internally
                    attr_real0 = target_c[:, :2].long(
                    )  # NO GENDER BINARIZATION
                    attr_real1 = target_c[:, 2:].view(
                        label_guided.size(0), label_guided[:, 2:].size(1) // 2,
                        2).max(dim=-1)[1]
                    attr_real = torch.cat([attr_real0, attr_real1], dim=1)

                    if self.mask:
                        attr_mask_fake = attnet_mask.calculate_attr_given_images(
                            sem_images)
                        attr_mask_real = attr_real
                        mean_sem = get_mean_for_iou(rec_images)
                        rec_iou = IoU(real_x, mean_sem)
                    else:
                        # import ipdb; ipdb.set_trace()
                        sem_rec_images = [
                            masknet(denorm(i), scatter=True)
                            for i in rec_images
                        ]
                        sem_rec_images = get_mean_for_iou(sem_rec_images)
                        rec_iou = IoU(mask, sem_rec_images)

                    if self.attr[idx] not in lpips_score.keys():
                        all_files[self.attr[idx]] = [files]
                        lpips_score[self.attr[idx]] = [lpips_value]
                        pose_score['yaw'][self.attr[idx]] = [yaw_error.cpu()]
                        pose_score['pitch'][self.attr[idx]] = [
                            pitch_error.cpu()
                        ]
                        pose_score['roll'][self.attr[idx]] = [roll_error.cpu()]
                        attr_score['real'][self.attr[idx]] = [attr_real.cpu()]
                        attr_score['fake'][self.attr[idx]] = [attr_fake.cpu()]
                        iou_score[self.attr[idx]] = [rec_iou]
                        if self.mask:
                            attr_mask_score['real'][self.attr[idx]] = [
                                attr_mask_real.cpu()
                            ]
                            attr_mask_score['fake'][self.attr[idx]] = [
                                attr_mask_fake.cpu()
                            ]

                    else:
                        all_files[self.attr[idx]].append(files)
                        lpips_score[self.attr[idx]].append(lpips_value)
                        pose_score['yaw'][self.attr[idx]].append(
                            yaw_error.cpu())
                        pose_score['pitch'][self.attr[idx]].append(
                            pitch_error.cpu())
                        pose_score['roll'][self.attr[idx]].append(
                            roll_error.cpu())
                        attr_score['real'][self.attr[idx]].append(
                            attr_real.cpu())
                        attr_score['fake'][self.attr[idx]].append(
                            attr_fake.cpu())
                        iou_score[self.attr[idx]].append(rec_iou)
                        if self.mask:
                            attr_mask_score['real'][self.attr[idx]].append(
                                attr_mask_real.cpu())
                            attr_mask_score['fake'][self.attr[idx]].append(
                                attr_mask_fake.cpu())

            # break
            # if count_img >= 20:
            #     # early debugging
            #     # import ipdb; ipdb.set_trace()
            #     break
        # import ipdb; ipdb.set_trace()
        # results = pickle.load(open(os.path.join(self.dir_name, 'scores_{}.pkl'.format(self.mode)), 'rb'))
        results = Munch(files=all_files,
                        lpips=lpips_score,
                        pose=pose_score,
                        attr=attr_score)
        results['iou_rec'] = iou_score
        if self.mask:
            results['attr_mask'] = attr_mask_score
        results_file = os.path.join(self.dir_name,
                                    'scores_{}.pkl'.format(self.mode))
        pickle.dump(results, open(results_file, 'wb'))
        # Replace binary GENDER with independent Male and Female labels
        parents_attr = list(dataset.parent_attrs.keys())[1:]
        parents_attr = ['Male', 'Female'] + parents_attr
        results.attr = compute_f1(results.attr, parents_attr)
        plot_PR_from_munch(results.attr, self.dir_name, parents_attr)
        if self.mask:
            results.attr_mask = compute_f1(results.attr_mask, parents_attr)
            plot_PR_from_munch(results.attr_mask,
                               self.dir_name,
                               parents_attr,
                               mask=True)
        return results
