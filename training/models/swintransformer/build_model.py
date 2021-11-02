## Pytorch implementation of Swin Transformer
## Adapted from: https://github.com/microsoft/Swin-Transformer/commit/6bbd83ca617db8480b2fb9b335c476ffaf5afb1a


from swin_transformer import SwinTransformer
# window size M =7 by default
# query dimension of each head is d = 32
# channel number of the hidder layers
# Swin-T: C = 96, layer numbers = [2, 2, 6, 2]  # complexity similar to ResNet-50
# Swin-S: C = 96, layer numbers = [2, 2, 18, 2] # complexity similar to ResNet-101
# Swin-B: C = 128, layer numbers = [2, 2, 18, 2]
# Swin-L: C = 192, layer numbers = [2, 2, 18, 2]

# sample config
config = {
    'feature_extractor': True,  # (bool): If True, drop last fc
    'img_size': 224,  # (int | tuple(int)): Input image size. Default 224
    'patch_size': 4,  # (int | tuple(int)): Patch size. Default: 4
    'in_chans': 3,  # (int): Number of input image channels. Default: 3
    'num_class': 1000,  # (int): Number of classes for classification head. Default: 1000
    'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
    'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer. (see above)
    'num_heads': [3, 6, 12, 24],  # (tuple(int)): Number of attention heads in different layers.
    'window_size': 7,  # (int): Window size. Default: 7
    'mlp_ratio': 4.0,  # (float): Ratio of mlp hidden dim to embedding dim. Default: 4
    'qkv_bias': True,  # (bool): If True, add a learnable bias to query, key, value. Default: True
    'qk_scale': None,  # (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
    'drop_rate': 0.0,  # (float): Dropout rate. Default: 0
    'drop_path_rate': 0.1,  # (float): Stochastic depth rate. Default: 0.1
    'ape': False,  # (bool): If True, add absolute position embedding to the patch embedding. Default: False
    'patch_norm': True,  # (bool): If True, add normalization after patch embedding. Default: True
    'use_checkpoint': False  # (bool): Whether to use checkpointing to save memory. Default: False
}

model = SwinTransformer(feature_extractor=config['feature_extractor'],
                        img_size=config['img_size'],
                        patch_size=config['patch_size'],
                        in_chans=config['in_chans'],
                        num_classes=config['num_class'],
                        embed_dim=config['embed_dim'],
                        depths=config['depths'],
                        num_heads=config['num_heads'],
                        window_size=config['window_size'],
                        mlp_ratio=config['mlp_ratio'],
                        qkv_bias=config['qkv_bias'],
                        qk_scale=config['qk_scale'],
                        drop_rate=config['drop_rate'],
                        drop_path_rate=config['drop_path_rate'],
                        ape=config['ape'],
                        patch_norm=config['patch_norm'],
                        use_checkpoint=config['use_checkpoint'])