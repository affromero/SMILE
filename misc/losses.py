# losses
import torch
import torch.nn as nn
import torch.nn.functional as F


def _perceptual_loss(model, img, target, config):
    from misc.utils import vgg_preprocess, to_cuda
    if config.IDENTITY_PERCEPTUAL == 'munit':
        IN = to_cuda(nn.InstanceNorm2d(512, affine=False))
        img_feat = IN(model(vgg_preprocess(img))[1])
        target_feat = IN(model(vgg_preprocess(target))[1])
        criterion = nn.MSELoss()
        cosine_distance = False

    elif config.IDENTITY_PERCEPTUAL == 'mtcnn':
        # Feat before classification
        img_feat = model(img)
        target_feat = model(target)
        criterion = nn.CosineSimilarity(dim=1, eps=1e-06)
        cosine_distance = True

    real_content = img_feat[-1]
    target_content = target_feat[-1]
    if cosine_distance:
        loss = 1 - criterion(real_content, target_content)
        if config.THRESHOLD_PERCEPTUAL:
            _loss = loss[loss > config.THRESHOLD_PERCEPTUAL]
            if _loss.nelement() == 0:
                loss /= 1000.  # Ignore it in practice
            else:
                loss = _loss
        loss = loss.mean()
    else:
        loss = criterion(real_content, target_content)
    return loss


# ==================================================================#
# ==================================================================#
def gan_feat_loss(real_feat, fake_feat):
    criterion = nn.L1Loss()
    from_f = 0
    # early activations could be different
    real_feat = real_feat[from_f:-1]
    fake_feat = fake_feat[from_f:-1]
    #  last feat is the output disc
    loss_feat = 0
    for r_pred, f_pred in zip(real_feat, fake_feat):
        loss_feat += criterion(r_pred, f_pred)
    loss_feat /= len(real_feat)
    return loss_feat


# ==================================================================#
# ==================================================================#
def get_disc(Disc, input, label, seg=None):
    dr1_input = None
    # if Disc.config.USE_MASK and not Disc.config.NO_PIX2PIX_DISC:
    #     input = torch.cat([input, seg], dim=1)
    #     src, gan_feat = Disc(input)
    # else:
    #     # import ipdb; ipdb.set_trace()
    #     src, dr1_input = Disc(input, label, sem=seg)
    #     gan_feat = None
    src = Disc(input, label, sem=seg)
    gan_feat = None
    # if not Disc.config.USE_MASK or Disc.config.NO_PIX2PIX_DISC:
    #     src = src[label == 1]
    if not isinstance(src, (tuple, list)):
        src = [src]
    return src, gan_feat


# ==================================================================#
# ==================================================================#
def _GAN_LOSS_FAKE(Disc, fake_x, label, seg=None, **kwargs):
    src_fake, _, _ = get_disc(Disc, fake_x, label, seg=seg)
    if Disc.config.HINGE:
        minval = torch.min(-src_fake - 1, get_zero_tensor(src_fake))
        loss_fake = -torch.mean(minval)
    else:
        zeros = torch.zeros_like(src_fake)
        loss_fake = F.binary_cross_entropy_with_logits(src_fake, zeros)
    return loss_fake


# ==================================================================#
# ==================================================================#
def _GAN_LOSS(Disc,
              real_x,
              fake_x,
              label,
              config,
              isFake=False,
              noR1=False,
              Epoch=None,
              fake_label=None,
              seg=None,
              fake_seg=None,
              feat_loss=False,
              idt=None,
              **kwargs):
    if not isFake:  # and (not Disc.config.USE_MASK or Disc.config.DR1):
        real_x.requires_grad_()
        seg.requires_grad_()
    src_real, gan_feat = get_disc(Disc, real_x, label, seg=seg)
    _input_disc = [real_x]
    # import ipdb; ipdb.set_trace()
    loss = {}
    ones = [torch.ones_like(s) for s in src_real]
    BCE_stable = torch.nn.BCEWithLogitsLoss()
    loss_real = 0
    loss_fake = 0
    for src, one in zip(src_real, ones):
        loss_real += BCE_stable(src, one)

    if isFake:
        loss['src'] = loss_real
        if gan_feat is not None and feat_loss:
            if 'real_feat' not in locals():
                _, real_feat = get_disc(Disc, fake_x, fake_label, seg=fake_seg)
            loss['feat'] = gan_feat_loss(real_feat, gan_feat)
    else:

        src_fake, _ = get_disc(Disc, fake_x, fake_label, seg=fake_seg)
        dr1 = 0
        for _inp, src in zip(_input_disc, src_real):
            dr1 += compute_grad2(src, _inp)
        loss['dr1'] = dr1
        zeros = [torch.zeros_like(s) for s in src_fake]
        for src, zero in zip(src_fake, zeros):
            loss_fake += BCE_stable(src, zero)
        loss['src'] = (loss_real + loss_fake) / 2.

    return loss


def get_zero_tensor(input):
    tensor = torch.FloatTensor(1).fill_(0).to(input)
    tensor.requires_grad_(False)
    return tensor.expand_as(input)


# ==================================================================#
# ==================================================================#
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(outputs=d_out.sum(),
                                    inputs=x_in,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    # import ipdb; ipdb.set_trace()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
