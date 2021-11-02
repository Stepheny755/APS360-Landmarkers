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
```