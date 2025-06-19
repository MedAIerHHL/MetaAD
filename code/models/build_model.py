from  .VAEGAN import create_generator, create_discriminator

def get_model(cfg, **kwargs):
    models = None
    if cfg.MODEL.name == 'VAEGAN':
        vae = create_generator(cfg, model_path=cfg.MODEL.vae_path, **kwargs)
        discriminator = create_discriminator(cfg, model_path=cfg.MODEL.discriminator_path, **kwargs)
        models = [vae, discriminator]
    return models