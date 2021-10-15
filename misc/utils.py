from __future__ import print_function


def scale_image(out, sem, org, size=48):
    from torch.nn import functional as F
    # import ipdb; ipdb.set_trace()
    _resize = size
    thickness = 2
    _out = out.clone()
    if sem is not None:
        _out[:, :, -_resize - (thickness * 2):, :_resize + (thickness * 2)] = 1
        _sem = F.interpolate(sem.clone(), (_resize, _resize), mode='nearest')
        _out[:, :, -_resize - thickness:-thickness, thickness:_resize +
             thickness] = _sem
    if org is not None:
        _out[:, :, -_resize - (thickness * 2):, -_resize -
             (thickness * 2):] = 1
        _org = F.interpolate(org.clone(), (_resize, _resize),
                             mode='bilinear',
                             align_corners=False)
        _out[:, :, -_resize - thickness:-thickness, -_resize -
             thickness:-thickness] = _org
    return _out


# ==================================================================#
# ==================================================================#
def cam2img(tensor, size=256):
    import numpy as np
    import cv2
    import torch.nn.functional as F
    import torch
    X = tensor.transpose(0, 1)  # class, bs, ...
    cam_img = []
    for bs_x in X:
        cam_img.append([])
        for x in bs_x:
            x = x[None, None, :, :]  # size x size x 1
            # x = x - np.min(x) # already minmax during forward
            # _cam_img = x / np.max(x)
            x = F.interpolate(x, size, mode='bilinear')
            x = x.detach().cpu().numpy()[0, 0][:, :, None]
            _cam_img = np.uint8(255 * x)
            # _cam_img = cv2.resize(_cam_img, (size, size), cv2.INTER_CUBIC)
            _cam_img = cv2.applyColorMap(_cam_img, cv2.COLORMAP_JET)
            _cam_img = cv2.cvtColor(_cam_img, cv2.COLOR_BGR2RGB)
            _cam_img = _cam_img.transpose(2, 0, 1)
            _cam_img = torch.from_numpy(_cam_img / 255.).type(type(tensor))
            cam_img[-1].append(_cam_img)
    cam_img = [torch.stack(i, 0) for i in cam_img]
    # cam_img = torch.stack(cam_img, 0) # class, bs, 3, size, size
    return cam_img


def minmax_heatmap(heatmap):
    _heatmap = heatmap.view(heatmap.size(0), -1)
    vector_min = _heatmap.min(dim=1, keepdim=True)[0]
    min_h = _heatmap.min(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    min_h = min_h.expand_as(heatmap)
    max_h = (_heatmap - vector_min).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
    max_h = max_h.unsqueeze(-1).expand_as(heatmap)
    _heatmap = (heatmap - min_h) / max_h
    # _heatmap = 1 - _heatmap # Invert colormap
    assert _heatmap.max() == 1.0 and _heatmap.min() == 0.0
    return _heatmap


# ==================================================================#
# ==================================================================#
def check_dir(dirname):
    import os
    if os.path.exists(dirname) and os.path.isdir(dirname):
        if not os.listdir(dirname):
            # print("Directory is empty")
            return False
        else:
            # print("Directory is not empty")
            return True
    else:
        # print("Given Directory don't exists")
        return False


# ==================================================================#
# ==================================================================#
def _CLS(self, model, data):
    import torch
    import torch.nn.functional as F
    data = to_cuda(data)
    out_label = model(data)[1]
    if len(out_label) > 1:
        out_label = torch.cat(
            [F.sigmoid(out.unsqueeze(-1)) for out in out_label],
            dim=-1).mean(dim=-1)
    else:
        out_label = F.sigmoid(out_label[0])
    out_label = (out_label > 0.5).float()
    return out_label


# ==================================================================#
# ==================================================================#
def circle_frame(tensor, thick=5, color='green', row_color=None):
    import numpy as np
    import torch
    from scipy import ndimage
    _color_ = {'green': (-1, -1, 1), 'red': (-1, -1, 1), 'blue': (-1, -1, 1)}
    _tensor = tensor.clone()
    size = tensor.size(2)
    thick = ((size / 2)**2) / 7.5
    xx, yy = np.mgrid[:size, :size]
    circle = (xx - size / 2)**2 + (yy - size / 2)**2
    donut = np.logical_and(circle < ((size / 2)**2 + thick), circle >
                           ((size / 2)**2 - thick))
    if color == 'blue':
        donut = ndimage.binary_erosion(donut, structure=np.ones(
            (15, 1))).astype(donut.dtype)
    elif color == 'red':
        donut = ndimage.binary_erosion(donut, structure=np.ones(
            (1, 15))).astype(donut.dtype)
    donut = np.expand_dims(donut, 0) * 1.
    donut = donut.repeat(tensor.size(1), axis=0)
    for i in range(donut.shape[0]):
        donut[i] = donut[i] * _color_[color][i]
    donut = to_cuda(torch.FloatTensor(donut))
    if row_color is None:
        row_color = [0, -1]
    else:
        row_color = [row_color]
    for nn in row_color:  # First and last frame
        _tensor[nn] = tensor[nn] + donut
    return _tensor


# ============================================================#
# ============================================================#
def color(dict, key, color='red'):
    from termcolor import colored
    dict[key] = colored('%.2f' % (dict[key]), color)


# ==================================================================#
# ==================================================================#
def compute_lpips(img0, img1, model=None, gpu=True):
    # RGB image from must be [-1,1]
    if model is None:
        from misc.lpips_model import DistModel
        model = DistModel()
        version = '0.0'  # Totally different values with 0.1
        model.initialize(model='net-lin',
                         net='alex',
                         use_gpu=gpu,
                         version=version)
    dist = model.forward(img0, img1)
    return dist, model


# ==================================================================#
# ==================================================================#
def config_yaml(config, yaml_file):
    def dict_dataset(dict):
        import os
        config.dataset = os.path.join(config.dataset, dict['dataset'])

    import yaml
    with open(yaml_file, 'r') as stream:
        config_yaml = yaml.load(stream)
    # if config.ALL_ATTR == 0:
    #     dict_dataset(config_yaml)
    for key, value in config_yaml.items():
        if 'ALL_ATTR_{}'.format(config.ALL_ATTR) in key:
            for key in config_yaml['ALL_ATTR_{}'.format(
                    config.ALL_ATTR)].keys():
                setattr(
                    config, key,
                    config_yaml['ALL_ATTR_{}'.format(config.ALL_ATTR)][key])
                if key == 'dataset':
                    dict_dataset(config_yaml['ALL_ATTR_{}'.format(
                        config.ALL_ATTR)])
        else:
            setattr(config, key, value)


# ==================================================================#
# ==================================================================#
def color_frame(tensor, thick=5, color='green', first=False):
    _color_ = {'green': (-1, 1, -1), 'red': (1, -1, -1), 'blue': (-1, -1, 1)}
    # tensor = to_data(tensor)
    for i in range(thick):
        for k in range(tensor.size(1)):
            # for nn in [0,-1]: #First and last frame
            for nn in [0]:  # First
                tensor[nn, k, i, :] = _color_[color][k]
                if first:
                    tensor[nn, k, :, i] = _color_[color][k]
                tensor[nn, k, tensor.size(2) - i - 1, :] = _color_[color][k]
                tensor[nn, k, :, tensor.size(2) - i - 1] = _color_[color][k]
    return tensor


# ==================================================================#
# ==================================================================#
def get_screen_name(gpu):
    import socket
    import os
    try:
        screen_name = os.environ['STY'].split('.')[-1]
        screen_name = 'screen_' + screen_name
    except KeyError:
        server_name = socket.gethostname()
        screen_name = '{}_gpu{}'.format(server_name, gpu)
    return screen_name


# ==================================================================#
# ==================================================================#
def copy_file_to_screen(file_txt, gpu='None'):
    import os
    dist = distributed()
    if dist.rank() != 0:
        return
    screen_txt = os.path.join('screen_files', '{}')
    file_txt = os.path.join('..', file_txt)
    screen_name = get_screen_name(gpu)
    screen_txt = screen_txt.format(screen_name)
    if os.path.islink(screen_txt):
        os.remove(screen_txt)
    os.symlink(file_txt, screen_txt)

    # copyfile(file_txt, screen_txt)
    # print(colored('Executing outside of Screen', 'red'))
    # print(colored('Press ENTER to continue', 'red'))
    # input("---")
    # return


# ==================================================================#
# ==================================================================#
def create_arrow(img_path, row=2, column=2, image_size=256):

    import cv2 as cv
    import numpy as np
    from PIL import Image, ImageFont, ImageDraw
    img = cv.imread(img_path)

    hskip = image_size // 8
    thickness = image_size // 64
    x_start = (image_size * column) + hskip
    x_end = (image_size * (column + 1)) - hskip
    y_pos = (image_size // 2) + (row * image_size)
    pointY = [y_pos, y_pos]
    pointX = [x_start, x_end]
    cv.arrowedLine(
        img,
        (pointX[0], pointY[0]),
        (pointX[1], pointY[1]),
        (0, 0, 0),
        thickness,
        #    tipLength=0.08,
    )

    cv.imwrite(img_path, img)


# ==================================================================#
# ==================================================================#
def create_text(img_path,
                text,
                size_text=8,
                row=2,
                column=2,
                image_size=256,
                rotate=0,
                background='white',
                force_replace=False):
    from PIL import ImageDraw, Image
    import cv2 as cv
    import numpy as np

    def get_font():
        from PIL import ImageFont
        return lambda size: ImageFont.truetype("data/Times-Roman.otf", size)

    foreground = (0, 0, 0) if background == 'white' else (255, 255, 255)
    background = (255, 255, 255) if background == 'white' else (0, 0, 0)

    # import ipdb; ipdb.set_trace()
    if force_replace:
        img_big = cv.imread(img_path)
        img = img_big[int(image_size * row):int(image_size *
                                                (row + 1)), image_size *
                      column:image_size * (column + 1)]
        img = Image.fromarray(img)
    else:
        base_size = (image_size, image_size)
        img_big = cv.imread(img_path)

        img = Image.new('RGB', base_size, background)

    text = text.split('\n')
    text = [line.capitalize() for line in text]

    draw = ImageDraw.Draw(img)

    font = get_font()(size_text)
    previous_y = 0
    for _text in text[::-1]:
        textsize = font.getsize(_text)
        textX = (img.size[0] - textsize[0]) / 2
        textY = img.size[1] - textsize[1] - previous_y - 5
        draw.text((textX, textY), _text, font=font, fill=foreground)
        previous_y += textsize[1]
    img = img.rotate(rotate)
    img = np.array(img)
    x_pos = [column * image_size, (column + 1) * image_size]
    y_pos = [int(row * image_size), int((row + 1) * image_size)]
    img_big[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1]] = img
    cv.imwrite(img_path, img_big)


# ==================================================================#
# ==================================================================#
def create_circle(image, size=256):
    import numpy as np
    import torch
    xx, yy = np.mgrid[:size, :size]
    # circles contains the squared distance to the (size, size) point
    # we are just using the circle equation learnt at school
    circle = (xx - size / 2)**2 + (yy - size / 2)**2
    bin_circle = (circle <= (size / 2)**2) * 1.
    bin_circle = torch.from_numpy(bin_circle).float()
    bin_circle = bin_circle.repeat(1, image.size(1), 1, image.size(-1) // size)
    image = (image * bin_circle) + (1 - bin_circle).clamp_(min=0, max=1)
    return image


# ==================================================================#
# ==================================================================#
def create_dir(dir):
    import os
    if '.' in os.path.basename(dir):
        dir = os.path.dirname(dir)
    if not os.path.isdir(dir):
        os.makedirs(dir)


# ==================================================================#
# ==================================================================#
def compress_image(filename, quality=50):
    from PIL import Image
    im = Image.open(filename)
    im.save(filename, "JPEG", quality=quality)


# ==================================================================#
# ==================================================================#
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def norm(x):
    out = (x * 2) - 1
    return out.clamp_(-1, 1)


# ==================================================================#
# ==================================================================#
def distributed():
    try:
        import torch.distributed
        assert torch.distributed.is_initialized()

        class dist():
            def __init__(self):
                self.horovod = False

            def init(self):
                pass

            def size(self):
                return torch.distributed.get_world_size()

            def rank(self):
                return torch.distributed.get_rank()

            def barrier(self):
                return torch.distributed.barrier()

            def bcast(self, tensor, root=0):
                return torch.distributed.broadcast(tensor, root=root)

            def is_available(self):
                return True

        _dist = dist()

    # except AssertionError:
    #     from mpi4py import MPI
    #     import horovod.torch as hvd
    #     class dist():
    #         def __init__(self)
    #             self.horovod = True
    #             self.hvd = hvd()

    #         def init(self):
    #             self.hvd.init()

    #         def size(self):
    #             return self.hvd.size()

    #         def rank(self):
    #             return self.hvd.rank()

    #         def barrier(self):
    #             comm = MPI.COMM_WORLD
    #             return comm.Barrier()

    #         def bcast(self, name, root=0):
    #             return comm.bcast(name, root=root)

    #     _dist = dist()

    except BaseException:

        class dist():
            def __init__(self):
                self.horovod = False

            def init(self):
                pass

            def size(self):
                return 1

            def rank(self):
                return 0

            def barrier(self):
                pass

            def bcast(self, name, root=0):
                return name

            def is_available(self):
                return False

        _dist = dist()

    return _dist


def distributed_horovod():
    from mpi4py import MPI
    import horovod.torch as hvd

    # try:
    class dist():
        def __init__(self):
            self.horovod = True
            self.hvd = hvd
            self.comm = MPI.COMM_WORLD

        def init(self):
            self.hvd.init()

        def size(self):
            return self.hvd.size()

        def rank(self):
            return self.hvd.rank()

        def barrier(self):
            return self.comm.Barrier()

        def bcast(self, name, root=0):
            return self.comm.bcast(name, root=root)

        def broadcast_parameters(self, *args, **kwargs):
            return self.hvd.broadcast_parameters(*args, **kwargs)

        def broadcast_optimizer_state(self, *args, **kwargs):
            return self.hvd.broadcast_optimizer_state(*args, **kwargs)

        def DistributedOptimizer(self, *args, **kwargs):
            return self.hvd.DistributedOptimizer(*args, **kwargs)

        def is_available(self):
            return True

        # _dist = dist()

    # except BaseException:

    #     class dist():
    #         def __init__(self):
    #             self.horovod = False

    #         def init(self):
    #             pass

    #         def size(self):
    #             return 1

    #         def rank(self):
    #             return 0

    #         def barrier(self):
    #             pass

    #         def bcast(self, name, root=0):
    #             return name

    #         def is_available(self):
    #             return False

    #     _dist = dist()

    return dist()


# ==================================================================#
# ==================================================================#
def elapsed_time(init_time):
    import time
    import datetime
    elapsed = time.time() - init_time
    return str(datetime.timedelta(seconds=elapsed)).split('.')[0]
    # Remove microseconds


# ==================================================================#
# ==================================================================#
def get_batch_debug(data_loader, batch_size, random_noise,
                    random_content=None):
    import torch
    fixed_x = []
    fixed_label = []
    fixed_mask = []
    deepfashion = data_loader.dataset.name == 'DeepFashion2'
    for i, data in enumerate(data_loader):
        images, labels = data['image'], data['label']
        mask = data['mask']
        fixed_x.append(images)
        fixed_label.append(labels)
        fixed_mask.append(mask)
        if i == max(1, 16 // batch_size):
            break
    fixed_x = torch.cat(fixed_x, dim=0)
    fixed_label = torch.cat(fixed_label, dim=0)
    fixed_mask = torch.cat(fixed_mask, dim=0)
    fixed_style = random_noise(fixed_x, seed=0)
    if random_content is not None:
        fixed_content = random_content(fixed_x, seed=0)
        fixed_style = (fixed_style, fixed_content)
    return fixed_x, fixed_label, fixed_style, fixed_mask


# ============================================================#
# ============================================================#
def get_fake(real_c,
             zero_pair=False,
             style_guided=False,
             c_dim=1,
             Random=False,
             seed=None):
    import torch
    if Random:
        fake_c = torch.randint_like(real_c, 0, 2)

    elif zero_pair and c_dim == 2:
        fake_c = 1 - real_c
        # For binary datasets, the identity will be avoided

    elif zero_pair:
        rand_idx = get_randperm(real_c, seed=seed)
        fake_c = real_c[rand_idx]
        for i in range(fake_c.size(0)):
            while True:
                index_real = real_c[i].tolist().index(1)
                index_fake = fake_c[i].tolist().index(1)
                # Zero pair between consecutive labels
                if abs(index_real - index_fake) > 1:
                    break
                rand_i = get_randperm(real_c[i])
                fake_c[i] = real_c[i][rand_i]

    elif style_guided:
        fake_c = real_c[torch.arange(real_c.size(0) - 1, -1, -1)]
        # rand_idx = get_randperm(real_c, seed=seed)
        fake_c = real_c[rand_idx]

    else:
        fake_c = real_c[torch.arange(real_c.size(0) - 1, -1, -1)]
        # rand_idx = get_randperm(real_c, seed=seed)
        # fake_c = real_c[rand_idx]

    return fake_c


# ==================================================================#
# ==================================================================#
def get_labels(image_size,
               dataset,
               style_guided=False,
               binary=False,
               colors=False,
               from_content=False,
               semantics=False,
               total_column=False,
               data_info=None):
    import imageio
    import glob
    import torch
    import skimage.transform
    import numpy as np
    from data.attr2img import external2img

    def imread(x):
        return imageio.imread(x)

    def resize(x):
        return skimage.transform.resize(x, (image_size, image_size))

    if dataset not in ['EmotioNet', 'BP4D']:
        selected_attrs = []
        config = data_info.config
        sec_attr = list(data_info.selected_attrs)
        if config.USE_MASK_TRAIN_ATTR and config.mode != 'train':
            sec_attr += list(data_info.semantic_attr.keys())
        if config.USE_MASK_TRAIN_SEMANTICS_ONLY:
            sec_attr = list(data_info.semantic_attr.keys())
        for attr in sec_attr:
            if colors:
                _attr = attr.split('Color_')[1]
                for color in data_info.selected_colors:
                    __attr = '{}_{}'.format(color, _attr)
                    selected_attrs.append(__attr)
                continue
            elif attr == 'Male' and not style_guided and not binary:
                # attr = 'Male/_Female'
                attr = 'Male'
            elif attr == 'Young' and not style_guided:
                # attr = 'Young/_Old'
                attr = 'Young'
            elif 'Hair' in attr:
                pass
            elif dataset == 'CelebA' and not style_guided:
                attr += '_Swap'
            else:
                pass
            if semantics:
                _attr = '{}_Mask'.format(attr)
                selected_attrs.append(_attr)
            selected_attrs.append(attr)
        labels = []
        if not from_content:
            labels += ['Source']
        if style_guided:
            labels += ['Target']
        labels += selected_attrs
        if total_column:
            labels += ['Total']
        imgs = external2img(labels, img_size=image_size)
        imgs = [resize(np.array(img)).transpose(2, 0, 1) for img in imgs]
    else:
        labels = []
        if not from_content:
            labels += ['Source']
        if style_guided:
            labels += ['Target']
        imgs = external2img(labels, img_size=image_size)
        imgs = [resize(np.array(img)).transpose(2, 0, 1) for img in imgs]
        imgs_file = sorted(glob.glob('data/{}/aus_flat/*g'.format(dataset)))
        imgs_file.pop(1)  # Removing 'off'
        imgs_file.pop(0)  # Removing 'Source'
        # import ipdb; ipdb.set_trace()
        imgs += [resize(imread(line)).transpose(2, 0, 1) for line in imgs_file]
    imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(
        np.float32)).unsqueeze(0)
    return imgs


# ==================================================================#
# ==================================================================#
def get_localhost():
    return 'http://129.132.67.120'


# ==================================================================#
# ==================================================================#
def get_loss_value(x):
    return x.detach().item()


# ============================================================#
# ============================================================#
def get_randperm(x, seed=None):
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    if x.size(0) > 2:
        rand_idx = to_cuda(torch.randperm(x.size(0)))
    elif x.size(0) == 2:
        rand_idx = to_cuda(torch.LongTensor([1, 0]))
    else:
        rand_idx = to_cuda(torch.LongTensor([0]))
    return rand_idx


# ==================================================================#
# ==================================================================#
def get_torch_version():
    import torch
    return float('.'.join(torch.__version__.split('.')[:2]))


# ==================================================================#
# ==================================================================#
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# ==================================================================#
# ==================================================================#
def handle_error(function):
    import time

    def decorator(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if (time.time() - args[0].start_time) > 600:
                # Error after 10 minutes of training
                body = 'Hey, there was an error!\n {}\n Fix it quickly!\n'
                body = body.format(str(e))
                body += unformated_text(args[0].Log)
                send_mail(body=body)
            raise e

    return decorator


# ==================================================================#
# ==================================================================#
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0:
        return str(num)
    else:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# ==================================================================#
# ==================================================================#
def imgShow(img, name='dummy'):
    from torchvision.utils import save_image
    try:
        save_image(denorm(img).cpu(), f'{name}.jpg')
    except BaseException:
        save_image(denorm(img.data).cpu(), f'{name}.jpg')


# ============================================================#
# ============================================================#
def imshow(img):
    import matplotlib.pyplot as plt
    from misc.utils import denorm, to_data
    img = to_data(denorm(img), cpu=True).numpy()
    img = img.transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()


# ============================================================#
# ============================================================#
def imshow_fromtensor(tensor, nrows=None):
    from torchvision.utils import save_image
    from PIL import Image
    import os
    nrows = max(tensor.size(0) // 3, 1) if nrows is None else nrows
    temp_file = 'dummy.jpg'
    save_image(tensor, temp_file, nrow=nrows, padding=0)
    Image.open(temp_file).show()
    os.remove(temp_file)


# ==================================================================#
# ==================================================================#
def interpolation(z1, z2, size):
    import torch
    import numpy as np
    z_interp = torch.FloatTensor(
        np.array([slerp(sz, z1, z2) for sz in np.linspace(0, 1, size)]))
    return z_interp


# ==================================================================#
# ==================================================================#
def save_json(json_file, filename):
    import json
    import os
    op = 'w'
    # if os.path.isfile(filename):
    #     op = 'a'
    # else:
    #     op = 'w'
    with open(filename, op) as f:
        json.dump(json_file,
                  f,
                  separators=(',', ': '),
                  indent=4,
                  sort_keys=False)


# ==================================================================#
# ==================================================================#
def load_inception(path='data/RafD/normal/inception_v3.pth'):
    from torchvision.models import inception_v3
    import torch
    import torch.nn as nn
    state_dict = torch.load(path)
    net = inception_v3(pretrained=False, transform_input=True)
    print("Loading inception_v3 from " + path)
    net.aux_logits = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
    net.load_state_dict(state_dict)
    for param in net.parameters():
        param.requires_grad = False
    return net


# ==================================================================#
# ==================================================================#
def make_gif(imgs, path, im_size=256, total_styles=5):
    import imageio
    import numpy as np
    if 'jpg' in path:
        path = path.replace('jpg', 'gif')
    imgs = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    target_size = (im_size, im_size, imgs.shape[-1])
    img_list = []
    for x in range(imgs.shape[2] // im_size):
        for bs in range(imgs.shape[0]):
            if x == 0 and bs > 1:
                continue  # Only save one image of the originals
            if x == 1:
                continue  # Do not save any of the 'off' label
            img_short = imgs[bs, :, im_size * x:im_size * (x + 1)]
            assert img_short.shape == target_size
            img_list.append(img_short)
    imageio.mimsave(path, img_list, duration=0.8)

    writer = imageio.get_writer(path.replace('gif', 'mp4'), fps=3)
    for im in img_list:
        writer.append_data(im)
    writer.close()


# ==================================================================#
# ==================================================================#
def mean_std_tensor(x):
    import torch
    import math
    if isinstance(x, (list, tuple)):
        x = [i for i in x if not math.isnan(i)]
        x = torch.FloatTensor(x)
    elif len(x.shape) == 1:
        pass
    elif math.isnan(x):
        return ('single_nan', 0)
    else:
        return (x, 0)
    mean = x.mean().item()
    std = x.std().item()
    return (mean, std)


# ==================================================================#
# ==================================================================#
def one_hot(labels, dim):
    """Convert label indices to one-hot vector"""
    import torch
    import numpy as np
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


# ==================================================================#
# ==================================================================#
def unformated_text(text):
    import re
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', str(text))


# ==================================================================#
# ==================================================================#
def PRINT(file, string):
    unformated_txt = unformated_text(string)
    if isinstance(file, str):
        with open(file, 'a') as f:
            print(unformated_txt, file=f)
            f.flush()
    elif file is not None:
        print(unformated_txt, file=file)
        file.flush()
    print(string)


# ==================================================================#
# ==================================================================#
def plot_txt(txt_file):
    import matplotlib.pyplot as plt
    lines = [line.strip().split() for line in open(txt_file).readlines()]
    legends = {idx: line
               for idx, line in enumerate(lines[0][1:])}  # 0 is epochs
    lines = lines[1:]
    epochs = []
    losses = {loss: [] for loss in legends.values()}
    for line in lines:
        epochs.append(line[0])
        for idx, loss in enumerate(line[1:]):
            losses[legends[idx]].append(float(loss))

    import pylab as pyl
    plot_file = txt_file.replace('.txt', '.pdf')
    _min = 4 if len(losses.keys()) > 9 else 3
    for idx, loss in enumerate(losses.keys()):
        # plot_file = txt_file.replace('.txt','_{}.jpg'.format(loss))

        plt.rcParams.update({'font.size': 10})
        ax1 = plt.subplot(3, _min, idx + 1)
        # err = plt.plot(epochs, losses[loss], 'r.-')
        err = plt.plot(epochs, losses[loss], 'b.-')
        plt.setp(err, linewidth=2.5)
        plt.ylabel(loss.capitalize(), fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        ax1.tick_params(labelsize=8)
        plt.hold(False)
        plt.grid()
    plt.subplots_adjust(left=None,
                        bottom=None,
                        right=None,
                        top=None,
                        wspace=0.5,
                        hspace=0.5)
    pyl.savefig(plot_file, dpi=100)


# ==================================================================#
# ==================================================================#
def pdf2png(filename):
    from wand.image import Image
    from wand.color import Color
    import os
    with Image(filename="{}.pdf".format(filename), resolution=500) as img:
        with Image(width=img.width,
                   height=img.height,
                   background=Color("white")) as bg:
            bg.composite(img, 0, 0)
            bg.save(filename="{}.png".format(filename))
    os.remove('{}.pdf'.format(filename))


# ==================================================================#
# ==================================================================#
def replace_weights(target, source, list):
    for l in list:
        target[l] = source[l]


# ==================================================================#
# ==================================================================#
def save_img(x, ncol, filename, denormalize=True):
    from torchvision.utils import save_image
    if denormalize:
        x = denorm(x)
    save_image(x.cpu(), filename, nrow=ncol, padding=0)


# ==================================================================#
# ==================================================================#
def send_mail(body="schusch",
              attach=[],
              subject='Message from schusch',
              to='roandres@ethz.ch'):
    import os
    content_type = {
        'jpg': 'image/jpeg',
        'gif': 'image/gif',
        'mp4': 'video/mp4'
    }
    if len(attach):  # Must be a list with the files
        enclosed = []
        for line in attach:
            format = line.split('.')[-1]
            enclosed.append('--content-type={} --attach {}'.format(
                content_type[format], line))
        enclosed = ' '.join(enclosed)
    else:
        enclosed = ''
    mail = 'echo "{}" | mail -s "{}" {} {}'.format(body, subject, enclosed, to)
    # print(mail)
    os.system(mail)


# ==================================================================#
# ==================================================================#
def single_source(tensor):
    import torch
    source = torch.ones_like(tensor)
    middle = 0  # int(math.ceil(tensor.size(0)/2.))-1
    source[middle] = tensor[0]
    return source


# ==================================================================#
# ==================================================================#
def slerp(val, low, high):
    """
  original: Animating Rotation with Quaternion Curves, Ken Shoemake
  https://arxiv.org/abs/1609.04468
  Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
  """
    import numpy as np
    omega = np.arccos(
        np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin(
        (1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


# ============================================================#
# ============================================================#
def split(data):
    # RaGAN uses different data for Dis and Gen
    try:
        if data.size(0) == 1:
            return data, data
        else:

            def split(x):
                if isinstance(x, (list, tuple)):
                    _len = len(x)
                else:
                    _len = x.size(0)
                return x[:_len // 2], x[_len // 2:]

            return split(data)

    except ValueError:
        return data, data


# ==================================================================#
# ==================================================================#
def target_debug_list(size, dim, config=None):
    import torch
    target_c_list = []
    for j in range(dim):
        target_c = torch.zeros(size, dim)
        target_c[:, j] = 1
        target_c_list.append(to_cuda(target_c))
    return target_c_list


# ==================================================================#
# ==================================================================#
def TimeNow():
    import datetime
    from tzlocal import get_localzone  # $ pip install tzlocal
    date_zone = get_localzone()
    return str(datetime.datetime.now(date_zone)).split('.')[0]


# ==================================================================#
# ==================================================================#
def isTimeWork():
    import datetime
    from tzlocal import get_localzone  # $ pip install tzlocal
    date_zone = get_localzone()
    date = datetime.datetime.now(date_zone)
    isWeekend = date.isoweekday() > 5
    isNight = date.hour >= 19 or date.hour <= 8
    if isNight or isWeekend:
        return False
    else:
        return True


# ==================================================================#
# ==================================================================#
def timeit(func):
    import functools
    import time

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('-> Function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))

    return newfunc


# ==================================================================#
# ==================================================================#
def TimeNow_str():
    import re
    return re.sub(r'\D', '_', TimeNow())


# ==================================================================#
# ==================================================================#
def to_cpu(x):
    return x.cpu() if x.is_cuda else x


# ==================================================================#
# ==================================================================#
def to_cuda(x, fixed=False):
    import torch
    import torch.nn as nn
    import os
    if x is None:
        return x

    class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
        # class DataParallel(nn.DataParallel):
        def __init__(self, model, device_ids=None):
            if device_ids is None:
                super().__init__(model)
            else:
                super().__init__(model, device_ids)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    class DataParallel(torch.nn.DataParallel):
        def __init__(self, model, device_ids=None):
            if device_ids is None:
                super().__init__(model)
            else:
                super().__init__(model, device_ids=device_ids)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(x, nn.Module):
        try:
            gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            gpu_ids = [0]
        # if distributed().size() > 1 and not distributed().horovod and not fixed:
        # import ipdb; ipdb.set_trace()
        if len(gpu_ids) > 1 and not distributed().horovod and not fixed:
            x = DataParallel(x, device_ids=[i for i in range(len(gpu_ids))])
            # x = DataParallel(x)
            x.to(device)
        elif distributed().size() > 1 and not distributed(
        ).horovod and not fixed:
            x.to(device)
            x = DistributedDataParallel(x)
        else:
            x.to(device)
        return x
    else:
        return x.to(device)


# ==================================================================#
# ==================================================================#
def to_data(x, cpu=False):
    x = x.data
    if cpu:
        x = to_cpu(x)
    return x


# ==================================================================#
# ==================================================================#
def to_numpy(x):
    return x.detach().cpu().numpy()


# ==================================================================#
# ==================================================================#
def to_parallel(main, input, list_gpu):
    import torch.nn as nn
    if len(list_gpu) > 1 and input.is_cuda:
        return nn.parallel.data_parallel(main, input, device_ids=list_gpu)
    else:
        return main(input)


# ==================================================================#
# ==================================================================#
def to_var(x, volatile=False, requires_grad=False, no_cuda=False):
    if not no_cuda:
        x = to_cuda(x)
    if get_torch_version() > 0.3:
        if requires_grad:
            return x.requires_grad_(True)
        elif volatile:
            return x.requires_grad_(False)
        else:
            return x

    else:
        from torch.autograd import Variable
        if isinstance(x, Variable):
            return x
        return Variable(x, volatile=volatile, requires_grad=requires_grad)


# ==================================================================#
# ==================================================================#
def vgg_preprocess(batch):
    import torch
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(to_cuda(mean))  # subtract mean
    return batch


# ==================================================================#
# ==================================================================#
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self,
                 localhost=get_localhost(),
                 port=8097,
                 env_name='main',
                 add_str=''):
        from visdom import Visdom
        import os
        import sys
        if env_name != 'main':
            self.log_file = os.path.join(env_name, 'log')
        else:
            self.log_file = env_name + '_log'
        if add_str:
            env_name = os.path.join(add_str, env_name)
        env_name = env_name.replace('/', '_')
        self.vis = Visdom(localhost,
                          port=port,
                          env=env_name,
                          log_to_filename=self.log_file)
        self.env = env_name
        self.remove_env(add_str)
        self.text('main command', ' '.join(sys.argv))
        self.plots = {}

    def remove_env(self, add_str):
        for env_list in self.vis.get_env_list():
            if env_list == 'main':
                self.vis.delete_env('main')
            elif add_str in env_list and self.env != env_list:
                self.vis.delete_env(env_list)

    def plot(self, x, y, var_name, split_name='train', title_name=''):
        import numpy as np
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                                 Y=np.array([y, y]),
                                                 env=self.env,
                                                 opts=dict(legend=[split_name],
                                                           title=title_name,
                                                           xlabel='Epochs',
                                                           ylabel=var_name))
        else:
            self.vis.line(X=np.array([x]),
                          Y=np.array([y]),
                          env=self.env,
                          win=self.plots[var_name],
                          name=split_name,
                          update='append')

    def text(self, name, text):
        self.vis.text(text, win=name)

    def image(self, image, var_name, title='', caption=''):
        self.vis.image(image,
                       win=var_name,
                       opts=dict(store_history=True,
                                 title=title,
                                 caption=caption,
                                 jpgquality=80))

    def images(self, image, var_name, title='', caption=''):
        self.vis.images(image,
                        win=var_name,
                        opts=dict(store_history=True,
                                  title=title,
                                  caption=caption,
                                  jpgquality=80))

    def add_scalar(self, var_name, y, global_step, **kwargs):
        x = global_step
        self.plot(x, y, var_name, **kwargs)

    def add_image(self, var_name, image, *args, **kwargs):
        self.image(image, var_name, **kwargs)

    def add_images(self, var_name, image, *args, **kwargs):
        self.images(image, var_name, **kwargs)
