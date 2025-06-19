import logging

from models.cyclegan.generator import Generator
from models.cyclegan.discriminator import Discriminator
from utils.tools import load_model, weights_init

# ----------------------------------------
#             Create Networks
# ----------------------------------------

def create_generator(cfg, model_path=None):
    generator = Generator(
        ngf                 = cfg.MODEL.feature_channels,
        n_res_blocks        = cfg.MODEL.num_res_blocks,
        channel_multipliers = cfg.MODEL.channel_mult,
        use_spectral_norm   = cfg.MODEL.spectral_norm,
        mask = cfg.MODEL.mask,
        mask_k = cfg.MODEL.mask_k,
        mask_prob = cfg.MODEL.mask_prob,
        mask_size = cfg.MODEL.mask_size,
    )
    logging.info('Generator is created!')

    # Initialize the networks
    if model_path is not None:
        generator = load_model(generator, model_path)
        print(f'Load pre-trained generator from {model_path}')
    else:
        weights_init(generator, init_type=cfg.MODEL.init_type, init_gain=cfg.MODEL.init_gain)
        logging.info('Initialize generator with %s type' % cfg.MODEL.init_type)

    return generator


def create_discriminator(cfg, model_path=None):
    discriminator = Discriminator(
        ndf               = cfg.MODEL.feature_channels,
        use_spectral_norm = cfg.MODEL.spectral_norm,
    )
    logging.info('Discriminator is created!')

    # Initialize the networks
    if model_path is not None:
        discriminator = load_model(discriminator, model_path)
        print(f'Load pre-trained discriminator from {model_path}')
    else:
        weights_init(discriminator, init_type=cfg.MODEL.init_type, init_gain=cfg.MODEL.init_gain)
        logging.info('Initialize discriminator with %s type' % cfg.MODEL.init_type)

    return discriminator