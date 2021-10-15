import os
import glob

__DATASETS__ = [
    os.path.basename(line).split('.py')[0]
    for line in glob.glob('datasets/*.py')
]


def base_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--dataset',
                        type=str,
                        default='CelebA_HQ',
                        choices=__DATASETS__)
    parser.add_argument('--dataset_test',
                        type=str,
                        default='',
                        choices=[''] + __DATASETS__)
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val', 'test', 'demo'])
    parser.add_argument('--mode_data',
                        type=str,
                        default='faces',
                        choices=['faces', 'normal'])
    parser.add_argument('--color_dim', type=int, default=3)
    parser.add_argument('--mask_dim', type=int, default=19)
    parser.add_argument('--conv_gen', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_size_test', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_sample', type=int, default=16)
    parser.add_argument('--gen_downscale', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--pretrained_mask', type=str, default='')
    parser.add_argument('--rgb_demo', type=str, default='')
    parser.add_argument('--rgb_label', type=str, default='')
    parser.add_argument('--sem_demo', type=str, default='')
    parser.add_argument('--ref_demo', type=str, default='')
    parser.add_argument('--keep_background',
                        action='store_true',
                        default=False)

    parser.add_argument('--seed', type=int, default=1)

    # Path
    parser.add_argument('--log_path', type=str, default='./snapshot/logs')
    parser.add_argument('--model_save_path',
                        type=str,
                        default='./snapshot/models')
    parser.add_argument('--sample_path',
                        type=str,
                        default='./snapshot/samples')
    parser.add_argument('--DEMO_PATH', type=str, default='')
    parser.add_argument('--DEMO_LABEL', type=str, default='')
    parser.add_argument('--json_file', type=str, default='results.json')

    # Generative
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--f_lr', type=float, default=1e-6)
    parser.add_argument('--lambda_src', type=float, default=1.0)
    parser.add_argument('--lambda_color', type=float, default=1.0)
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_ds', type=float,
                        default=20.0)  # very high for semantic masks
    parser.add_argument('--lambda_sty', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--lambda_rec_src', type=float, default=1.0)
    parser.add_argument('--lambda_idt', type=float, default=1.0)
    parser.add_argument('--leaky_relu', type=float, default=0.2)
    parser.add_argument('--seg_size',
                        type=int,
                        default=32,
                        choices=[16, 32, 64, 128])
    parser.add_argument('--noise_dim', type=int, default=16, choices=[16])
    parser.add_argument('--small_dim', type=int, default=16, choices=[16, 32])
    parser.add_argument('--style_dim', type=int, default=64, choices=[64, 128])
    parser.add_argument('--style_dim_rgb',
                        type=int,
                        default=64,
                        choices=[64, 128, 256, 512, 1024])
    parser.add_argument('--disc_dim',
                        type=int,
                        default=512,
                        choices=[512, 1024])
    parser.add_argument('--trunc_w_psi', type=float, default=0.8)

    parser.add_argument('--content_dim',
                        type=int,
                        default=64,
                        choices=[64, 512, 1024])

    # Misc
    parser.add_argument('--DELETE', action='store_true', default=False)
    parser.add_argument(
        '--ATTR',
        type=str,
        default='',
        choices=['gender', 'eyeglasses', 'bangs', 'smile', 'hair', 'single'])
    parser.add_argument('--TRAIN_MASK', action='store_true', default=False)
    parser.add_argument('--USE_MASK', action='store_true', default=False)
    parser.add_argument('--DS', action='store_true', default=False)
    parser.add_argument('--ORG_DS', action='store_true', default=True)
    parser.add_argument('--SN', action='store_true', default=False)
    parser.add_argument('--MIXED_REG', action='store_true', default=False)
    parser.add_argument('--MOD', action='store_true', default=False)
    # parser.add_argument('--GEN_NO_SHORTCUT', action='store_true', default=True)
    parser.add_argument('--CONTENT_GEN', action='store_true', default=False)
    parser.add_argument('--REC_CONTENT', action='store_true', default=False)
    parser.add_argument('--RANDOM_LABELS', action='store_true', default=False)
    parser.add_argument('--FIXED_LABELS', action='store_true', default=False)
    parser.add_argument('--IDENTITY', action='store_true', default=False)
    parser.add_argument('--ANTI_ALIAS', action='store_true', default=False)
    parser.add_argument('--FeatLoss', action='store_true', default=False)
    parser.add_argument('--UGATIT_DISC', action='store_true', default=False)
    parser.add_argument('--RAGAN', action='store_true', default=False)
    parser.add_argument('--NO_DETACH', action='store_true', default=False)
    parser.add_argument('--STARGAN_TRAINING',
                        action='store_true',
                        default=False)
    parser.add_argument('--RAP', action='store_true', default=False)
    parser.add_argument('--RAP_WEIGHT', action='store_true', default=False)
    parser.add_argument('--REPLICATE_MOD', action='store_true', default=False)
    parser.add_argument('--NOT_GENDER', action='store_true', default=False)
    parser.add_argument('--EARRINGS', action='store_true', default=False)
    parser.add_argument('--BANGS', action='store_true', default=False)
    parser.add_argument('--SMILE', action='store_true', default=False)
    parser.add_argument('--HAT', action='store_true', default=False)
    parser.add_argument('--SHORT_HAIR', action='store_true', default=False)
    parser.add_argument('--RGB_SEMANTICS', action='store_true', default=False)
    parser.add_argument('--UGATIT_DISC_CLS',
                        action='store_true',
                        default=False)
    parser.add_argument('--STYLE_SEMANTICS',
                        action='store_true',
                        default=False)
    parser.add_argument('--STYLE_SEMANTICS_ATTR',
                        action='store_true',
                        default=False)
    parser.add_argument('--STYLE_SEMANTICS_ALL',
                        action='store_true',
                        default=False)
    parser.add_argument('--NO_SEMANTIC_IDENTITY',
                        action='store_true',
                        default=False)
    parser.add_argument('--GENERAL_HEATMAP',
                        action='store_true',
                        default=False)
    parser.add_argument('--FAN', action='store_true', default=False)
    parser.add_argument('--TANH', action='store_true', default=False)
    parser.add_argument('--GroupNorm', action='store_true', default=False)
    parser.add_argument('--FILL_RGB', action='store_true', default=False)
    parser.add_argument('--WeightStandarization',
                        action='store_true',
                        default=False)
    parser.add_argument('--SEAN_IN', action='store_true', default=False)
    parser.add_argument('--SEAN_NOISE', action='store_true', default=False)
    parser.add_argument('--DISC_NOISE', action='store_true', default=False)
    parser.add_argument('--DEEPSEE', action='store_true', default=False)
    parser.add_argument('--GRAD_CLIPPING', action='store_true', default=False)
    parser.add_argument('--SMALL_DISC', action='store_true', default=False)
    parser.add_argument('--SMALL_DISC2', action='store_true', default=False)
    parser.add_argument('--NO_SHORTCUT', action='store_true', default=False)
    parser.add_argument('--DISC_SEAN', action='store_true', default=False)
    parser.add_argument('--SMALL_ENCODER', action='store_true', default=False)
    parser.add_argument('--HAIR', action='store_true', default=False)
    parser.add_argument('--GENDER', action='store_true', default=False)
    parser.add_argument('--EYEGLASSES', action='store_true', default=False)
    parser.add_argument('--IN', action='store_true', default=False)
    parser.add_argument('--DR1', action='store_true', default=False)
    parser.add_argument('--HINGE', action='store_true', default=False)
    parser.add_argument('--BETA_SEAN', action='store_true', default=False)
    parser.add_argument('--IDENTITY_FT', action='store_true', default=False)
    parser.add_argument('--MASK_DIVERSITY', action='store_true', default=False)
    parser.add_argument('--ATTR_DIVERSITY', action='store_true', default=False)
    parser.add_argument('--VGG_PERCEPTUAL', action='store_true', default=False)
    parser.add_argument('--IDENTITY_MASK', action='store_true', default=False)
    parser.add_argument('--WEIGHT_STYLE', action='store_true', default=False)
    parser.add_argument('--IDENTITY_FEAT', action='store_true', default=False)
    parser.add_argument('--STYLE_NOISE', action='store_true', default=False)
    parser.add_argument('--RESHAPE_MASK', action='store_true', default=False)
    parser.add_argument('--GENERATOR_512', action='store_true', default=False)
    parser.add_argument('--REENACTMENT', action='store_true', default=False)
    parser.add_argument('--SPLIT_STYLE', action='store_true', default=False)
    parser.add_argument('--SPLIT_STYLE2', action='store_true', default=False)
    parser.add_argument('--REPLICATE_ENCODER',
                        action='store_true',
                        default=False)
    parser.add_argument('--REPLICATE_DISC', action='store_true', default=False)
    parser.add_argument('--ENCODER_UNET', action='store_true', default=False)
    parser.add_argument('--MOD_RELU', action='store_true', default=False)
    parser.add_argument('--SKIP_CONNECTION',
                        action='store_true',
                        default=False)
    parser.add_argument('--USE_MASK_TRAIN_ATTR',
                        action='store_true',
                        default=False)
    parser.add_argument('--USE_MASK_TRAIN_SEMANTICS_MISSING',
                        action='store_true',
                        default=False)
    parser.add_argument('--USE_MASK_TRAIN_SEMANTICS_MISSING_WITH_HAIR',
                        action='store_true',
                        default=False)
    parser.add_argument('--USE_MASK_TRAIN_SEMANTICS_ONLY',
                        action='store_true',
                        default=False)
    parser.add_argument('--USE_MASK_TRAIN_SEMANTICS_NO_EYEGLASSES',
                        action='store_true',
                        default=False)

    parser.add_argument('--REMOVING_MASK', action='store_true', default=False)
    parser.add_argument('--ONLY_GEN', action='store_true', default=False)
    parser.add_argument('--NO_PIX2PIX_DISC',
                        action='store_true',
                        default=False)
    parser.add_argument('--NO_PIX2PIX_DISC2',
                        action='store_true',
                        default=False)
    parser.add_argument('--BINARY_CHILDREN',
                        action='store_true',
                        default=False)
    parser.add_argument(
        '--IDENTITY_PERCEPTUAL',
        type=str,
        choices=['', 'mtcnn', 'mtcnn2', 'mtcnn3', 'mtcnn3_in', 'munit'],
        default='')

    parser.add_argument('--DATASET_SAMPLED',
                        type=int,
                        default=100,
                        choices=[10, 20, 30, 50, 100])
    parser.add_argument('--ALL_ATTR', type=int, default=0)
    parser.add_argument('--CACHE', action='store_true', default=False)
    parser.add_argument('--TENSORBOARD', action='store_true', default=False)
    parser.add_argument('--VISDOM', action='store_true', default=False)
    parser.add_argument('--HOROVOD', action='store_true', default=False)
    parser.add_argument('--DISTRIBUTED', action='store_true', default=False)
    parser.add_argument('--GPU', type=str, default='-1')

    parser.add_argument('--local_rank', type=int, default=-1)

    # Scores
    parser.add_argument('--EVAL', action='store_true', default=False)

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument('--sample_step', type=int, default=500)
    # parser.add_argument('--model_save_step', type=int, default=10000)

    # Debug options
    parser.add_argument('--n_interpolation', type=int, default=5)
    parser.add_argument('--save_epoch',
                        type=int,
                        default=1,
                        help='Consider small datasets for epochs')
    parser.add_argument('--model_epoch',
                        type=int,
                        default=5,
                        help='Save weights each ** epochs')
    parser.add_argument('--sample_epoch',
                        type=int,
                        default=1,
                        help='Generate samples each ** epochs')
    parser.add_argument('--sample_iter',
                        type=int,
                        default=-1,
                        help='Generate samples each ** iterations')
    parser.add_argument('--log_epoch',
                        type=int,
                        default=1,
                        help='Save tensorboard logs each ** epochs')
    parser.add_argument('--style_debug', type=int, default=8)
    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--style_train_debug', type=int, default=12)
    parser.add_argument('--style_label_debug',
                        type=int,
                        default=2,
                        choices=[0, 1, 2, 3])
    config = parser.parse_args()
    return config
