from misc.visualization import save_video as SaveVideo
from solver import Solver
import warnings
import numpy as np
from misc.utils import denorm
import os
from misc.utils import TimeNow_str, to_cuda, norm
from misc.visualization import vis_parsing_maps, bisenet2sean
from misc.mask_utils import scatterMask, label2mask, label2mask_plain
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import io
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import random
from metrics.attribute_model import AttNet
from misc.utils import scale_image, create_arrow, create_text
from facenet_pytorch import MTCNN
warnings.filterwarnings('ignore')

_FORMAT_ = ['.png', '.jpg']


def Glob(dirname):
    from glob import glob
    files = []
    for _type in _FORMAT_:
        files += glob(os.path.join(dirname, '*' + _type))
    files = sorted(files)
    return files


class get_face(object):
    def __init__(self, image_size=256):
        self.mtcnn = MTCNN(image_size=image_size, margin=image_size // 2)
        # self.mtcnn = MTCNN(image_size=image_size, margin=image_size // 2.5)
        # self.mtcnn = MTCNN(image_size=image_size, margin=image_size // 3)
        # self.mtcnn = MTCNN(image_size=image_size, margin=image_size // 5)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            out = []
            for _x in x:
                out.append(self(transforms.ToPILImage()(_x)))
            out = torch.stack(out, dim=0)
            return out
        elif isinstance(x, Image.Image):
            return denorm(self.mtcnn(x))
        else:
            raise TypeError(f"Only receive Tensor or Image type, not {x.type}")


class Demo(Solver):
    def __init__(self, args, data_loader):
        super().__init__(args, data_loader)
        self.image_size = self.args.image_size_test
        self.init_seg()
        self.init_sean()
        self.get_face = get_face(args.image_size)
        self.n_domains = self.args.n_domains

        self.normalize = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform_rgb = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
        ])
        self.transform_sem = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.rgb_video = False
        if os.path.isdir(self.args.rgb_demo):
            self.rgb_demo = Glob(self.args.rgb_demo)
        elif os.path.isfile(
                self.args.rgb_demo) and 'mp4' not in self.args.rgb_demo:
            self.rgb_demo = [self.args.rgb_demo]
        elif 'mp4' in self.args.rgb_demo:
            self.rgb_demo = self.read_video(self.args.rgb_demo)[0]
            self.rgb_video = True
        else:
            raise TypeError("You must enter a valid rgb file or directory.")

        if self.args.rgb_label:
            self.rgb_label = []
            for f in self.args.rgb_label.split(','):
                self.rgb_label.extend([1 - float(f), float(f)])
            self.rgb_label = torch.FloatTensor(self.rgb_label)
        else:
            self.rgb_label = None

        self.run_bisenet = False

        if os.path.isdir(self.args.sem_demo):
            self.sem_demo = Glob(self.args.sem_demo)
        elif os.path.isfile(self.args.sem_demo):
            self.sem_demo = [self.args.sem_demo]
        else:
            self.run_bisenet = True

        self.attnet = to_cuda(AttNet())

    def init_seg(self):
        from metrics.segmentation_model import BiSeNet
        self.seg_model = BiSeNet(n_classes=19)
        file_pth = 'metrics/segmentation_weights.pth'
        weights = torch.load(file_pth,
                             map_location=lambda storage, loc: storage)
        self.seg_model.load_state_dict(weights)
        self.seg_model = to_cuda(self.seg_model)
        self.seg_model.eval()
        self.bisenet_transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Resize((512, 512), interpolation=Image.ANTIALIAS),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def init_sean(self):
        from metrics.smileSYN import SMILE_SYN
        self.sean = SMILE_SYN(self.image_size)

    def get_sample_ref(self, path='', files=False):
        if path == '':
            path = self.args.ref_demo
        if not os.path.isdir(path) and not os.path.isfile(path):
            raise TypeError(f"{path} not valid or not found.")
        print('Getting reference samples from', path)

        if os.path.isdir(path):
            imgs = sorted(os.listdir(path))
            img_path = [os.path.join(path, i) for i in imgs]
            img_path = [
                i for i in img_path if '.pt' not in i and 'project' not in i
            ]
        else:
            img_path = [path]
        proj_size_name = '' if self.image_size == 256 else '-{}'.format(
            self.image_size)
        projection = [
            os.path.splitext(i)[0] + proj_size_name + '.pt' for i in img_path
        ]
        # projection = [os.path.splitext(i)[0] + '-no_wplus.pt' for i in img_path]
        projection = [i for i in projection if os.path.isfile(i)]
        imgs = [Image.open(i).convert('RGB') for i in img_path]
        imgs = [transforms.ToPILImage()(self.get_face(i)) for i in imgs]
        imgs = [self.transform_rgb(i).unsqueeze(0) for i in imgs]
        imgs = to_cuda(torch.cat(imgs, dim=0))
        sem = self.img2seg(imgs)
        sem = scatterMask(sem[:, 0])
        imgs = (imgs - 0.5) * 2
        if len(img_path) == len(projection) and self.sean.projection:
            print('Using projection styles.')
            style, noise = [], []
            for idx, proj in enumerate(projection):
                metaproj = torch.load(
                    proj, map_location=lambda storage, loc: storage)
                # In case there is a missing style
                if 'no_wplus' in proj:
                    metaproj['latent'] = metaproj['latent'].unsqueeze(0)
                _style = self.sean.replace_w_with_seg(sem[idx].unsqueeze(0),
                                                      metaproj['latent'])
                style.append(_style)
                noise.append([to_cuda(i) for i in metaproj['noise']])
            style = to_cuda(torch.stack(style, 0))
        else:
            print('WITHOUT projection styles.')
            style = self.sean.model.extract_style(imgs, sem)
            noise = None
        if files:
            return imgs, sem, style, noise, img_path
        else:
            return imgs, sem, style, noise

    @torch.no_grad()
    def manipulation_ref(self, string=TimeNow_str()):
        last_name = self.resume_name()
        save_folder = os.path.join(self.args.sample_path,
                                   '{}_demo'.format(last_name))
        # string = TimeNow_str()
        save_folder = os.path.join(save_folder, string)
        os.makedirs(save_folder, exist_ok=True)
        self.nets_ema.G.eval()
        self.nets_ema.F.eval()
        self.nets_ema.S.eval()
        self.PRINT('Demo files will be saved in "{}"..!'.format(save_folder))

        video = []

        n_styles = 3
        seed_sean_style = [
            random.randint(0, 100000) for i in range(n_styles + 1)
        ]
        img_ref, sem_ref, sty_ref, noise_ref = self.get_sample_ref()
        if self.args.REENACTMENT:
            label = torch.ones((img_ref.size(0), 1)).to(img_ref.device)
        else:
            label = self.attnet(img_ref, one_hot=True)[-1]
            label[11, 2] = 0
            label[11, 3] = 1
            label[5, 2] = 0
            label[5, 3] = 1
            if not (label.sum(1) == 6).all().item():
                # only one image get fixed
                idx = (label.sum(1) == 6).tolist().index(False)
                # Gender confusion
                label[idx, 1] = 1
        style_ref_smile = self.nets_ema.S(sem_ref, label)

        save_name = os.path.join(save_folder, 'out_ref')
        os.makedirs(save_name, exist_ok=True)
        for count, filename in tqdm(enumerate(self.rgb_demo),
                                    total=len(self.rgb_demo)):

            if not self.rgb_video:
                img = self.read_img_from_file(filename,
                                              self.transform_rgb,
                                              norm=True).unsqueeze(0)
                img_big = img
            else:
                img = filename
                img = self.normalize(img).unsqueeze(0)
                img_big = img.clone()
                img = F.interpolate(img, (self.image_size, self.image_size),
                                    mode='bilinear')
            img_big = to_cuda(img_big)
            img = to_cuda(img)

            if self.args.FAN:
                if self.args.REENACTMENT:
                    fan = self.nets_ema.FAN.get_heatmap(img_ref)
                else:
                    fan = self.nets_ema.FAN.get_heatmap(img)
            else:
                fan = None

            if self.run_bisenet:
                img_big = denorm(img_big)
                sem = self.img2seg(img_big)
            else:
                sem = self.read_img_from_file(
                    filename, self.transform_sem)[0].unsqueeze(0)
                sem = to_cuda(sem)
            output_seg = []
            output_sean = []
            sem_input = scatterMask(sem[:, 0])
            if self.rgb_label is None:
                if self.args.REENACTMENT:
                    label_org = torch.ones((img.size(0), 1)).to(img.device)
                else:
                    label_org = self.attnet(img, one_hot=True)[1]
                    if not (label_org.sum(1) == 6).all().item():
                        # Gender confusion
                        label_org[0, 1] = 1
            else:
                label_org = self.rgb_label.unsqueeze(0).to(sem.device)
            style_org_smile = self.nets_ema.S(sem_input, label_org)

            _style_ref_smile = torch.cat((style_org_smile, style_ref_smile),
                                         dim=0)
            sem_input = sem_input.repeat(_style_ref_smile.size(0), 1, 1, 1)
            if self.args.REENACTMENT:
                fake_seg = self.nets_ema.G(sem_ref,
                                           style_org_smile.repeat(
                                               sem_ref.size(0), 1, 1),
                                           fan=fan)
            else:
                fake_seg = self.nets_ema.G(sem_input,
                                           _style_ref_smile,
                                           fan=fan)
            sean_input = scatterMask(label2mask_plain(fake_seg)[:, 0])
            fake_seg = label2mask(fake_seg)

            # Reconstruction
            output_sean_ref = []
            if not img_ref.size(0) == 16 and not self.args.REENACTMENT:
                sean_random = self.sean.forward_from_tensor(
                    sean_input[0].unsqueeze(0),
                    rgb_guide=img,
                    sem_guide=sean_input[0].unsqueeze(0),
                )
                sean_i = label2mask(sean_input[0].unsqueeze(0),
                                    n=sean_input[0].unsqueeze(0).size(1))
                sean_random = scale_image(sean_random, sean_i, None, size=64)
                output_sean_ref.append(sean_random)

            if not self.args.REENACTMENT:
                sean_input = sean_input[1:]

            # Reference styles
            for sean_i, img_r, sem_r, sty_r in zip(sean_input, img_ref,
                                                   sem_ref, sty_ref):
                sean_i = sean_i.unsqueeze(0)
                sty_r = sty_r.unsqueeze(0)
                img_r = img_r.unsqueeze(0)
                sem_r = sem_r.unsqueeze(0)
                sean_random = self.sean.forward_from_tensor(
                    sean_i,
                    rgb=img,
                    rgb_guide=img_r,
                    sem_guide=sem_r,
                    style_ref=sty_r,
                )
                sean_i = label2mask(sean_i, n=sean_i.size(1))
                sean_random = scale_image(sean_random, sean_i, None, size=64)
                output_sean_ref.append(sean_random)
            output_sean_ref = torch.cat(output_sean_ref, dim=-1)

            _img_ref = scale_image(denorm(img_ref),
                                   label2mask(sem_ref, n=sem_ref.size(1)),
                                   None,
                                   size=64)
            _img_ref = torch.cat([i.unsqueeze(0) for i in _img_ref],
                                 dim=-1).cpu()
            if img_ref.size(0) == 16:
                out1, out2 = torch.chunk(output_sean_ref, 2, dim=-1)
                _img_ref1, _img_ref2 = torch.chunk(_img_ref, 2, dim=-1)
                row1 = torch.cat([torch.ones_like(img).cpu(),
                                  _img_ref1.cpu()],
                                 dim=-1)
                row2 = torch.cat([torch.ones_like(img).cpu(),
                                  out1.cpu()],
                                 dim=-1)
                row3 = torch.cat([torch.ones_like(img).cpu(),
                                  out2.cpu()],
                                 dim=-1)
                row4 = torch.cat([torch.ones_like(img).cpu(),
                                  _img_ref2.cpu()],
                                 dim=-1)
                all_images_org = torch.cat([row1, row2, row3, row4], dim=-2)
                _img = scale_image(denorm(img),
                                   label2mask(scatterMask(sem[:, 0]),
                                              n=sem_input.size(1)),
                                   None,
                                   size=64)
                all_images_org[:, :,
                               img.size(-1) * 2 -
                               img.size(-1) // 2:img.size(-1) * 2 +
                               img.size(-1) // 2, :img.size(-1)] = _img.cpu()
                save_name_rgb = os.path.join(save_name,
                                             '%s.png' % (str(count).zfill(6)))
                save_image(all_images_org, save_name_rgb, nrow=1, padding=0)
                create_text(save_name_rgb,
                            'Reference Image',
                            size_text=img.size(-1) // 8,
                            rotate=90,
                            row=0,
                            column=0,
                            force_replace=True)
                create_text(save_name_rgb,
                            'Reference Image',
                            size_text=img.size(-1) // 8,
                            rotate=90,
                            row=3,
                            column=0,
                            force_replace=True)
                create_text(save_name_rgb,
                            'Input Video',
                            size_text=img.size(-1) // 8,
                            rotate=0,
                            row=0.5,
                            column=0,
                            force_replace=True)
            else:
                _img_ref = torch.cat([i.unsqueeze(0) for i in img_ref],
                                     dim=-1).cpu()
                _img_ref = torch.cat([denorm(img), _img_ref], dim=-1)
                output_seg = torch.cat([i.unsqueeze(0) for i in fake_seg],
                                       dim=-1).cpu()
                row1 = torch.cat([torch.ones_like(img).cpu(),
                                  _img_ref.cpu()],
                                 dim=-1)
                row2 = torch.cat([denorm(img).cpu(),
                                  output_sean_ref.cpu()],
                                 dim=-1)
                all_images_org = torch.cat([row1, row2], dim=-2)
                save_name_rgb = os.path.join(save_name,
                                             '%s.png' % (str(count).zfill(6)))
                save_image(all_images_org, save_name_rgb, nrow=1, padding=0)
                create_text(save_name_rgb,
                            'Reference Image',
                            size_text=img.size(-1) // 8,
                            rotate=90,
                            row=0,
                            column=0,
                            force_replace=True)
                create_text(save_name_rgb,
                            'Input Video',
                            size_text=img.size(-1) // 8,
                            rotate=0,
                            row=0,
                            column=0,
                            force_replace=True)
            # break
            # if count == 4:
            #     break
        if len(os.listdir(save_name)) > 1:
            imgs = Glob(save_name)
            imgs = [np.array(Image.open(i)) for i in imgs]
            imgs = np.stack(imgs, axis=0)
            save_name_video = os.path.join(save_folder, 'out_ref.mp4')
            SaveVideo(save_name_video,
                      imgs,
                      output_fps=15,
                      vcodec='libx264',
                      filters='')
            # ffmpeg_str = "ffmpeg -r 15 -i {}/%06d.png -vcodec libx264 -y {} -hide_banner"
            # save_name_video = os.path.join(save_folder, 'out_ref.mp4')
            # os.system(ffmpeg_str.format(save_name, save_name_video))

    # ==================================================================#
    # ==================================================================#
    @torch.no_grad()
    def __call__(self):
        string = TimeNow_str()
        self.manipulation_ref(string=string)

    def read_img_from_file(self, filename, transform, norm=False):
        img = Image.open(filename).convert('RGB')
        img = self.get_face(img)
        if norm:
            img = self.normalize(img)
        return img

    @torch.no_grad()
    def img2seg(self, tensor, color=True):
        if not hasattr(self, 'seg_model'):
            self.init_seg()
        assert tensor.min() >= 0 and tensor.max() <= 1
        tensor = F.interpolate(tensor, (512, 512), mode='bilinear')
        tensor = [self.bisenet_transform(t).unsqueeze(0) for t in tensor]
        tensor = torch.cat(tensor, dim=0)
        seg = self.seg_model(tensor)[0]
        seg = F.interpolate(seg, (self.image_size, self.image_size),
                            mode='bilinear')
        parsing = torch.argmax(seg, dim=1, keepdims=True)
        parsing = bisenet2sean(parsing)
        return parsing

    def get_color_sem(self, rgb, seg):
        seg_numpy = seg.cpu().numpy()
        rgb_numpy = rgb.cpu().numpy().transpose(1, 3)
        sem_color = []
        mix_color = []
        for image, parsing in zip(rgb_numpy, seg_numpy):
            color_sem, mix_sem = vis_parsing_maps(image, parsing, stride=1)
            sem_color.append(color_sem)
            mix_color.append(mix_sem)
        sem_color = np.array(sem_color).transpose(1, 3)
        mix_color = np.array(mix_color).transpose(1, 3)
        return rgb_numpy, seg_numpy, sem_color, mix_color

    def read_video(self, video_file):
        # Please only read rgb video files,
        # Semantic videos are normally pixel-wisely corrupted
        vid_frames, _, metadata = io.read_video(video_file)
        vid_frames = vid_frames.transpose(1, 3).transpose(2, 3) / 255.
        vid_frames = vid_frames[::5, :, 400:-100, 70:-100]
        return vid_frames, metadata

    def save_video(self, dirname, dict):  # rgb=None, sem=None):
        assert 'rgb' in dict.keys() or 'sem' in dict.keys()
        for key, value in dict.items():
            _dirname = dirname + '_' + key
            os.makedirs(_dirname, exist_ok=True)
            for i, frame in enumerate(value):
                Image.fromarray(frame).save(
                    os.path.join(_dirname, '{}.png'.format(str(i).zfill(4))))


# python main.py --batch_size=8 --GPU=3 --FAN --ATTR=Gender --TRAIN_MASK
# --STYLE_SEMANTICS --ORG_DS --mode=demo --rgb_demo out1.mp4 --MOD

# python main.py --batch_size=4 --GPU=NO_CUDA --FAN --EYEGLASSES --GENDER --EARRINGS --HAT --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS --lambda_ds=20 --MOD --SPLIT_STYLE --mode=demo --rgb_demo out1.mp4
# python main.py --batch_size=4 --GPU=NO_CUDA --FAN --EYEGLASSES --GENDER
# --HAT --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS --lambda_ds=10 --MOD
# --SPLIT_STYLE --mode=demo --rgb_demo out2.mp4 --rgb_label 1,0,0,0

# REENACTMENT
# python main.py --batch_size=4 --GPU=NO_CUDA --image_size=512 --FAN
# --REENACTMENT --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS --lambda_ds=20
# --MOD --mode=demo --rgb_demo Figures/out3.mp4 --ref_demo
# Figures/ffhq_sample --ATTR single

# python main.py --batch_size=4 --GPU=NO_CUDA --FAN --EYEGLASSES --GENDER
# --HAT --EARRINGS --HAIR --BANGS --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS
# --lambda_ds=20 --MOD --SPLIT_STYLE

# python main.py --batch_size=4 --GPU=NO_CUDA --FAN --EYEGLASSES --GENDER
# --EARRINGS --HAT --BANGS --HAIR --ORG_DS --TRAIN_MASK --STYLE_SEMANTICS
# --lambda_ds=20 --MOD --SPLIT_STYLE --mode=demo --ref_demo
# Figures/ffhq_teaser --rgb_demo Figures/teaser_input.png
