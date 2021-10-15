def update_folder(config, folder):
    import os
    if config.pretrained_model:
        return
    config.log_path = os.path.join(config.log_path, folder)
    config.sample_path = os.path.join(config.sample_path, folder)
    config.model_save_path = os.path.join(config.model_save_path, folder)


def remove_folder(config):
    import os
    folder = os.path.join(config.sample_path, '*.jpg')
    folder += ' ' + os.path.join(config.sample_path, '*.txt')
    folder += ' ' + os.path.join(config.model_save_path, '*.pth')
    # folder += ' ' + os.path.join(config.log_path, 'events*')
    folder += ' ' + os.path.join(config.log_path, 'log')
    folder += ' ' + os.path.join(config.sample_path, '*eval*')
    folder += ' ' + os.path.join(config.sample_path, '*val*')
    folder += ' ' + os.path.join(config.sample_path, '*_test')
    folder += ' ' + os.path.join(config.sample_path, '*json')
    print("YOU ARE ABOUT TO REMOVE EVERYTHING IN:\n{}".format(
        folder.replace(' ', '\n')))
    input("ARE YOU SURE?")
    os.system("rm -rf {}".format(folder))


def UPDATE_FOLDER(config, str):
    if getattr(config, str):
        update_folder(config, str)


def update_config(config):
    import os
    import glob

    update_folder(config, config.dataset)
    if '/' in config.dataset:
        config.dataset = config.dataset.split('/')[0]
    config.num_epochs *= config.save_epoch
    config.num_epochs_decay *= config.save_epoch

    if config.image_size != 256:
        update_folder(config, 'image_size_' + str(config.image_size))
    if config.DATASET_SAMPLED != 100:
        update_folder(config, 'DATASET_SAMPLED_' + str(config.DATASET_SAMPLED))

    # config.BIAS = True
    if config.ORG_DS:
        config.DS = False

    if config.DS:
        config.ORG_DS = False

    if config.dataset == 'painters_14' and config.ATTR == '':
        config.ATTR = 'all'
    elif config.ATTR not in ['']:
        update_folder(config, 'ATTR_' + str(config.ATTR))
    # elif not config.TRAIN_MASK:
    # elif config.STYLE_SEMANTICS or config.dataset == 'DeepFashion2':
    # elif config.dataset != 'DeepFashion2':
    # else:
    #     config.BINARY_CHILDREN = True

    UPDATE_FOLDER(config, 'RGB_SEMANTICS')
    UPDATE_FOLDER(config, 'DS')
    UPDATE_FOLDER(config, 'ORG_DS')
    UPDATE_FOLDER(config, 'CONTENT_GEN')
    UPDATE_FOLDER(config, 'MIXED_REG')
    UPDATE_FOLDER(config, 'MOD')
    UPDATE_FOLDER(config, 'REPLICATE_MOD')
    UPDATE_FOLDER(config, 'REC_CONTENT')
    # UPDATE_FOLDER(config, 'BINARY_CHILDREN')
    UPDATE_FOLDER(config, 'ANTI_ALIAS')
    UPDATE_FOLDER(config, 'UGATIT_DISC')
    UPDATE_FOLDER(config, 'UGATIT_DISC_CLS')
    UPDATE_FOLDER(config, 'SN')
    UPDATE_FOLDER(config, 'IDENTITY')
    UPDATE_FOLDER(config, 'FIXED_LABELS')
    UPDATE_FOLDER(config, 'GENERAL_HEATMAP')
    UPDATE_FOLDER(config, 'TRAIN_MASK')
    UPDATE_FOLDER(config, 'USE_MASK')
    UPDATE_FOLDER(config, 'STYLE_SEMANTICS')
    UPDATE_FOLDER(config, 'STYLE_SEMANTICS_ATTR')
    UPDATE_FOLDER(config, 'STYLE_SEMANTICS_ALL')
    UPDATE_FOLDER(config, 'NO_SEMANTIC_IDENTITY')
    UPDATE_FOLDER(config, 'FAN')
    UPDATE_FOLDER(config, 'FeatLoss')
    UPDATE_FOLDER(config, 'TANH')
    UPDATE_FOLDER(config, 'SEAN_IN')
    UPDATE_FOLDER(config, 'DISC_SEAN')
    UPDATE_FOLDER(config, 'HINGE')
    UPDATE_FOLDER(config, 'NO_PIX2PIX_DISC')
    UPDATE_FOLDER(config, 'NO_PIX2PIX_DISC2')
    UPDATE_FOLDER(config, 'SEAN_NOISE')
    UPDATE_FOLDER(config, 'DISC_NOISE')
    UPDATE_FOLDER(config, 'DEEPSEE')
    UPDATE_FOLDER(config, 'GRAD_CLIPPING')
    UPDATE_FOLDER(config, 'SMALL_DISC')
    UPDATE_FOLDER(config, 'SMALL_DISC2')
    UPDATE_FOLDER(config, 'NO_SHORTCUT')
    UPDATE_FOLDER(config, 'IN')
    UPDATE_FOLDER(config, 'DR1')
    UPDATE_FOLDER(config, 'SMALL_ENCODER')
    UPDATE_FOLDER(config, 'BETA_SEAN')
    UPDATE_FOLDER(config, 'IDENTITY_FT')
    UPDATE_FOLDER(config, 'USE_MASK_TRAIN_ATTR')
    UPDATE_FOLDER(config, 'USE_MASK_TRAIN_SEMANTICS_MISSING')
    UPDATE_FOLDER(config, 'USE_MASK_TRAIN_SEMANTICS_MISSING_WITH_HAIR')
    UPDATE_FOLDER(config, 'USE_MASK_TRAIN_SEMANTICS_ONLY')
    UPDATE_FOLDER(config, 'USE_MASK_TRAIN_SEMANTICS_NO_EYEGLASSES')
    UPDATE_FOLDER(config, 'REMOVING_MASK')
    UPDATE_FOLDER(config, 'ONLY_GEN')
    UPDATE_FOLDER(config, 'RAGAN')
    UPDATE_FOLDER(config, 'STARGAN_TRAINING')
    UPDATE_FOLDER(config, 'NO_DETACH')
    UPDATE_FOLDER(config, 'GroupNorm')
    UPDATE_FOLDER(config, 'WeightStandarization')
    UPDATE_FOLDER(config, 'HAIR')
    UPDATE_FOLDER(config, 'RAP')
    UPDATE_FOLDER(config, 'RAP_WEIGHT')
    UPDATE_FOLDER(config, 'MASK_DIVERSITY')
    UPDATE_FOLDER(config, 'ATTR_DIVERSITY')
    UPDATE_FOLDER(config, 'IDENTITY_MASK')
    UPDATE_FOLDER(config, 'VGG_PERCEPTUAL')
    UPDATE_FOLDER(config, 'WEIGHT_STYLE')
    UPDATE_FOLDER(config, 'IDENTITY_FEAT')
    UPDATE_FOLDER(config, 'STYLE_NOISE')
    UPDATE_FOLDER(config, 'RESHAPE_MASK')
    UPDATE_FOLDER(config, 'GENDER')
    UPDATE_FOLDER(config, 'EYEGLASSES')
    UPDATE_FOLDER(config, 'NOT_GENDER')
    UPDATE_FOLDER(config, 'EARRINGS')
    UPDATE_FOLDER(config, 'HAT')
    UPDATE_FOLDER(config, 'SHORT_HAIR')
    UPDATE_FOLDER(config, 'BANGS')
    UPDATE_FOLDER(config, 'SMILE')
    UPDATE_FOLDER(config, 'REPLICATE_ENCODER')
    UPDATE_FOLDER(config, 'REPLICATE_DISC')
    UPDATE_FOLDER(config, 'ENCODER_UNET')
    UPDATE_FOLDER(config, 'MOD_RELU')
    UPDATE_FOLDER(config, 'SKIP_CONNECTION')
    UPDATE_FOLDER(config, 'GENERATOR_512')
    UPDATE_FOLDER(config, 'REENACTMENT')
    UPDATE_FOLDER(config, 'SPLIT_STYLE')
    UPDATE_FOLDER(config, 'SPLIT_STYLE2')

    if config.STARGAN_TRAINING:
        config.num_epochs //= 2
        config.num_epochs_decay //= 2

    # if config.mode == 'train':
    #     config.batch_size *= 2 # Split data during training

    if config.REENACTMENT:
        config.GENERAL_HEATMAP = True

    if config.RAP_WEIGHT:
        config.RAP = True

    if config.USE_MASK:
        config.ONLY_GEN = True
        config.STYLE_SEMANTICS = True
        config.FAN = True

    if config.dataset == 'Cityscapes':
        config.batch_sample = 9

    if config.FAN:
        config.gen_downscale = 5

    if config.ONLY_GEN:
        assert config.USE_MASK

    if config.USE_MASK_TRAIN_SEMANTICS_MISSING_WITH_HAIR:
        config.USE_MASK_TRAIN_SEMANTICS_MISSING = True

    if config.USE_MASK_TRAIN_SEMANTICS_MISSING:
        config.USE_MASK_TRAIN_ATTR = True

    if config.IDENTITY_FT:
        config.IDENTITY = True
        config.FeatLoss = True

    if config.BETA_SEAN:
        config.beta1 = 0.5
        config.beta2 = 0.999

    if config.SMALL_DISC2:
        config.SMALL_DISC = True

    if config.NO_PIX2PIX_DISC2:
        config.NO_PIX2PIX_DISC = True

    if config.TRAIN_MASK or config.USE_MASK:
        config.MASK = True
    else:
        config.MASK = False

    if config.STYLE_SEMANTICS_ALL:
        config.STYLE_SEMANTICS = True

    if config.seg_size != 32:
        update_folder(config, 'seg_size_' + str(config.seg_size))

    if config.small_dim != 16 and config.SPLIT_STYLE:
        update_folder(config, 'small_dim_' + str(config.small_dim))

    # if config.gen_downscale != 4:
    #     update_folder(config, 'gen_downscale_' + str(config.gen_downscale))

    if config.IDENTITY_PERCEPTUAL:
        update_folder(
            config,
            'IDENTITY_PERCEPTUAL_{}_{}'.format(config.IDENTITY_PERCEPTUAL,
                                               config.lambda_perceptual))

    # if config.mapping_lr != 1e-6:
    #     update_folder(config, 'mapping_lr_' + str(config.mapping_lr))

    if config.lambda_ds != 30.0 and config.TRAIN_MASK and config.ORG_DS:
        update_folder(config, 'lambda_ds_' + str(config.lambda_ds))

    if config.lambda_sty != 1.0:
        update_folder(config, 'lambda_sty_' + str(config.lambda_sty))

    if config.lambda_rec != 1.0:
        update_folder(config, 'lambda_rec_' + str(config.lambda_rec))

    if config.conv_gen != 32:
        update_folder(config, 'conv_gen_' + str(config.conv_gen))

    # if config.disc_dim != 1024:
    #     update_folder(config, 'disc_dim_' + str(config.disc_dim))

    if config.style_dim != 64:
        update_folder(config, 'style_dim_' + str(config.style_dim))

    if config.style_dim_rgb != 64 and config.USE_MASK and config.ORG_DS:
        update_folder(config, 'style_dim_rgb_' + str(config.style_dim_rgb))

    # if config.CONTENT_GEN:
    #     config.REC_CONTENT = True

    if config.image_size_test == 0:
        config.image_size_test = config.image_size

    if config.mode_data != 'faces':
        update_folder(config, 'data_' + config.mode_data)

    if config.DELETE:
        remove_folder(config)

    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.sample_path, exist_ok=True)

    if config.pretrained_model == '':
        try:
            config.pretrained_model = sorted(
                glob.glob(os.path.join(config.model_save_path, '*_G.pth')))[-1]
            config.pretrained_model = '_'.join(
                os.path.basename(config.pretrained_model).split('_')[:-1])
        except BaseException:
            pass

    if config.EVAL:
        config.mode = 'test'

    if config.mode == 'train':
        assert not (config.TENSORBOARD
                    and config.VISDOM), 'Only supported one of them at a time'
        config.loss_plot = os.path.abspath(
            os.path.join(config.sample_path, 'loss.txt'))
        config.log = os.path.abspath(
            os.path.join(config.sample_path, 'log.txt'))

        config.json_file = {
            'FID':
            os.path.join(config.sample_path, 'val_fid.json'),
            'LPIPS':
            os.path.join(config.sample_path, 'val_lpips.json'),
            'PERCEPTUAL_PRECISION':
            os.path.join(config.sample_path, 'val_precision.json'),
            'PERCEPTUAL_RECALL':
            os.path.join(config.sample_path, 'val_recall.json'),
            'PERCEPTUAL_DENSITY':
            os.path.join(config.sample_path, 'val_density.json'),
            'PERCEPTUAL_COVERAGE':
            os.path.join(config.sample_path, 'val_coverage.json'),
        }

    return config
