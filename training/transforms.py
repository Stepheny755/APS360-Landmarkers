import torchvision.transforms as transforms

class ColorDistortion:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        color_jitter = transforms.ColorJitter(0.8 * self.distortion, 0.8 * self.distortion,
                                              0.8 * self.distortion, 0.2 * self.distortion)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=1.0)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            # rnd_gray
        ])
        transformed_image = color_distort(image)
        return transformed_image

def get_transforms(
    dataset: str,
    color_aug: str = 'Color-Distortion',
    dist_factor: float = 0.3,
    degrees: float = 45.0,
    horizontal_flipping: float = 0.5,
    horizontal_shift: float = 0.1,
    vertical_shift: float = 0.1):

    if 'GLRv2' in dataset:
        if color_aug == 'Color-Distortion':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flipping),
                transforms.RandomAffine(degrees=degrees, 
                                        translate=(horizontal_shift, 
                                                vertical_shift)),
                ColorDistortion(dist_factor),
                transforms.ToTensor(),
                transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                transforms.CenterCrop(224),
                transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                transforms.CenterCrop(224),
                transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                transforms.CenterCrop(224),
                transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                transforms.CenterCrop(224),
                transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

    return train_transform, test_transform