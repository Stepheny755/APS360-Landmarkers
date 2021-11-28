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
from .models.swintransformer.swin_transformer import SwinTransformer

swin_config = {
'feature_extractor': False,  # (bool): If True, drop last fc
'img_size': 224,  # (int | tuple(int)): Input image size. Default 224
'num_class': num_classes,  # (int): Number of classes for classification head. Default: 1000
'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer. (see above)
'num_heads': [4, 8, 16, 32],  # (tuple(int)): Number of attention heads in different layers.
}

model = SwinTransformer(feature_extractor=swin_config['feature_extractor'],
    img_size=swin_config['img_size'],
    num_classes=swin_config['num_class'],
    embed_dim=swin_config['embed_dim'],
    depths=swin_config['depths'],
    num_heads=swin_config['num_heads'])

features = model(img)  # features shape [samples, 768]
```
