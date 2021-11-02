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