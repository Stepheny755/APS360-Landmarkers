import os
from torchvision.datasets import ImageFolder

class GLRv2(ImageFolder):
    folder_name = "landmark-recognition-1k"
    def __init__(self, root, transform=None):
        super(GLRv2, self).__init__(os.path.join(root, self.folder_name), transform)

