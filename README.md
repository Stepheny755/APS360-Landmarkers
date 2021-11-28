# APS360-Landmarkers

## Feature Extractor Models
### SENet
Adapted from: https://github.com/moskomule/senet.pytorch/releases/tag/archive

Example: 
```python
from src.senet.se_resnet import se_resnet50
model = se_resnet50(num_classes=1_000, feature_extractor=True)  # Load a SE_ResNet50 model as feature extractor
features = model(img)  # features shape [samples, 2048]
```

### EfficientNet
Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch/releases/tag/1.0

Example:
```python
from src.efficientnet.efficient_net import EfficientNet
model = EfficientNet.from_name('efficientnet-b0', override_params={'feature_extractor':True})  # Load an EfficientNet model as feature extractor
features = model(img) # features shape [samples, 1280]
```

### Swin-Transformer
Adapted from: https://github.com/microsoft/Swin-Transformer/commit/6bbd83ca617db8480b2fb9b335c476ffaf5afb1a

Example: see sample code in src/swintransformer/build_model.py
```python
# features shape [samples, 768]
from .models.swintransformer.swin_transformer import SwinTransformer

swin_config = {
            'feature_extractor': True,  # (bool): If True, drop last fc
            'img_size': 224,  # (int | tuple(int)): Input image size. Default 224
            'patch_size': 4,  # (int | tuple(int)): Patch size. Default: 4
            'in_chans': 3,  # (int): Number of input image channels. Default: 3
            'num_class': num_classes,  # (int): Number of classes for classification head. Default: 1000
            'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
            'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer. (see above)
            'num_heads': [4, 8, 16, 32],  # (tuple(int)): Number of attention heads in different layers.
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
        model = SwinTransformer(feature_extractor=swin_config['feature_extractor'],
                        img_size=swin_config['img_size'],
                        patch_size=swin_config['patch_size'],
                        in_chans=swin_config['in_chans'],
                        num_classes=swin_config['num_class'],
                        embed_dim=swin_config['embed_dim'],
                        depths=swin_config['depths'],
                        num_heads=swin_config['num_heads'],
                        window_size=swin_config['window_size'],
                        mlp_ratio=swin_config['mlp_ratio'],
                        qkv_bias=swin_config['qkv_bias'],
                        qk_scale=swin_config['qk_scale'],
                        drop_rate=swin_config['drop_rate'],
                        drop_path_rate=swin_config['drop_path_rate'],
                        ape=swin_config['ape'],
                        patch_norm=swin_config['patch_norm'],
                        use_checkpoint=swin_config['use_checkpoint'])
```
