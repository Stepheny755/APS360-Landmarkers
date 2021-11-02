import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # Suggested default setting
    parser.add_argument('--save_freq', type=int, default=25,
                        help='save frequency')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--p', type=int, default=1,
                        help='options: 1, 2')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='options: SGD, AdamP, Adam')
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='options: StepLR, MultiStepLR')
    parser.add_argument('--step_size', type=int, default=15,
                        help='period of learning rate decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=[10, 20, 70, 100],
                        help='Epoch indices separated by a comma')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # Dataset
    parser.add_argument('--dataset', type=str, default='GLM',
                        help='options: ADP-Release1')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to the parent containing the dataset folder')

    # network
    parser.add_argument('--network', type=str, default='resnet18',
                        help='network to train')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint to load')

    # Loss function and its arguments
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        choices=['CrossEntropyLoss'], help='choose loss function')

    # Augmentation methods and its arguments
    parser.add_argument('--color_augmentation', type=str, default='White-Balance',
                        choices=['White-Balance', 'Color-Distortion'],
                        help='choose a color augmentation method')
    parser.add_argument('--distortion_factor', type=float, default=0.5,
                        help='choose a distortion factor for the color augmentation; range - (0, 1]')

    config = parser.parse_args()

    # set the path to save the trained models
    config.model_path = './save/{}_models'.format(config.dataset)
    config.eval_folder = './eval/lr={}_step_size_{}_momentum_{}_decay_rate_{}'.\
        format(config.learning_rate, config.step_size,
               config.momentum, config.lr_decay_rate)
    config.model_name = '{}_{}_{}_lr_{}_decay_{}_step_size_{}'. \
        format(config.loss, config.dataset, config.network, config.learning_rate,
               config.lr_decay_rate, config.step_size)

    config.save_folder = os.path.join(config.model_path, config.model_name)
    if not os.path.isdir(config.save_folder):
        os.makedirs(config.save_folder)

    if not os.path.isdir(config.eval_folder):
        os.makedirs(config.eval_folder)

    return config