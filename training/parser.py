"""
definition for parsing functions for initiating code
"""
import argparse
import os

def parse_option():
    """
    Parser for command line calls of the train function
    """
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

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='options: SGD, AdamP, Adam')
    parser.add_argument('--scheduler', type=str, default='None',
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
    parser.add_argument('--dataset', type=str, default='GLRv2_5',
                        choices=['GLRv2', "GLRv2_5"],
                        help='Pick a dataset, default: GLRv2_5')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to the parent containing the dataset folder')

    # network
    parser.add_argument('--network', type=str, default='efficientnet-b0',
                        choices=['efficientnet-b0', "senet", "swin", "ResNet+SVM"],
                        help='network to train')
    parser.add_argument('--from_pretrained', type=str, default='True',
                        choices=['True', "False"],
                        help='whether the model is pretrained or not')
    # parser.add_argument('--checkpoint_path', type=str, default=None,
    #                     help='path to checkpoint to load')
    parser.add_argument('--freeze_layers', type=str, default="True",
                        choices=['True', "False"],
                        help='freeze all layers except last')

    # Loss function and its arguments
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        choices=['CrossEntropyLoss'], help='choose loss function')

    # Augmentation methods and its arguments
    parser.add_argument('--color_augmentation', type=str, default='Color-Distortion',
                        choices=['White-Balance', 'Color-Distortion', 'None'],
                        help='choose a color augmentation method')
    parser.add_argument('--distortion_factor', type=float, default=0.3,
                        help='choose a distortion factor for the color augmentation; range - (0, 1]')

    config = parser.parse_args()

    # set the path to save the trained models
    config.model_path = './save/{}_models'.format(config.dataset)
    config.eval_folder = './eval'
    config.model_name = '{}_{}_{}_lr_{}_decay_{}_step_size_{}'. \
        format(config.loss, config.dataset, config.network, config.learning_rate,
               config.lr_decay_rate, config.step_size)

    config.save_folder = os.path.join(config.model_path, config.model_name)
    os.makedirs(config.save_folder, exist_ok=True)
    os.makedirs(config.eval_folder, exist_ok=True)

    return config
