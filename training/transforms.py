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

    if color_aug == 'Color-Distortion':
        color_aug = ColorDistortion(dist_factor)
    else:
        color_aug = None

    if 'GLRv2' in dataset:
        train_transform_list = [
            transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]

        test_transform_list = [
            transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
        if "preprocessed" in dataset:
            if color_aug != None:
                train_transform_list = [
                    color_aug,
                    transforms.RandomHorizontalFlip(p=horizontal_flipping),
                    transforms.RandomAffine(degrees=degrees, 
                                            translate=(horizontal_shift, 
                                                    vertical_shift)),
                    transforms.ToTensor(),
                ] + train_transform_list

                test_transform_list = [
                    transforms.ToTensor(),
                ] + test_transform_list
            else:
                train_transform_list = [
                    transforms.RandomHorizontalFlip(p=horizontal_flipping),
                    transforms.RandomAffine(degrees=degrees, 
                                            translate=(horizontal_shift, 
                                                    vertical_shift)),
                    transforms.ToTensor(),
                ] + train_transform_list

                test_transform_list = [
                    transforms.ToTensor(),
                ] + test_transform_list
        else:
            if color_aug != None:
                train_transform_list = [
                    color_aug,
                    transforms.RandomHorizontalFlip(p=horizontal_flipping),
                    transforms.RandomAffine(degrees=degrees, 
                                            translate=(horizontal_shift, 
                                                    vertical_shift)),
                    transforms.ToTensor(),
                    transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                    transforms.CenterCrop(224),
                ] + train_transform_list

                test_transform_list = [
                    transforms.ToTensor(),
                    transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                    transforms.CenterCrop(224),
                ] + test_transform_list
            else:
                train_transform_list = [
                    transforms.RandomHorizontalFlip(p=horizontal_flipping),
                    transforms.RandomAffine(degrees=degrees, 
                                            translate=(horizontal_shift, 
                                                    vertical_shift)),
                    transforms.ToTensor(),
                    transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                    transforms.CenterCrop(224),
                ] + train_transform_list

                test_transform_list = [
                    transforms.ToTensor(),
                    transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
                    transforms.CenterCrop(224),
                ] + test_transform_list

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    return train_transform, test_transform