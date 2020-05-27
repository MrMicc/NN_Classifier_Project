# Train a Neural Network using transfer learning:
import argparse
import torch
import matplotlib; matplotlib.use('agg')
from torch import optim
from utils import data_loader, nn_model

# 0. Creating parser for the input values
parser = argparse.ArgumentParser(description="UDACity AI project")

# 1. Get the directory to the image files to train with
parser.add_argument('--data-directory', default='./flowers', help='Path to the image files. The folder shold contain: '
                                                                  '"train", "test" and "valid" folders')

# 2. Set the directory to save checkpoints
parser.add_argument('--save-dir', default='./checkpoints', help='Path to save the model that you have trained')

# 3. Choose the architecture
parser.add_argument('--arch', default='densenet121', help='Arch with the pre-trained model. In this version we support '
                                                          'densenet121 and vgg19')

# 4. Set the hyperparameters
# 4.1 set learning rate
parser.add_argument('--learning-rate', type=float, default='0.0003', help='Learning rate of the model that you are going'
                                                                          ' to train')

# 4.2 set droupout
parser.add_argument('--dropout', type=float, default='0.005', help='Dropout rate')

# 4.3 set hidden layer
parser.add_argument('--hidden-layers', nargs='+', default='256', type=int, help='weights of the hidden layers that are '
                                                                                'going to use. This version suport "N" layers')

# 4.4 set epochs
parser.add_argument('--epoch', type=int, default='5', help='Number of epoch the train mode are going to use')

# 4.4 set batch size
parser.add_argument('--batch-size', type=int, default='32', help='Batch size')

# parsing arguments
args = parser.parse_args()

data_directory = args.data_directory
save_directory = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
dropout = args.dropout
hidden_layers = args.hidden_layers
epoch = args.epoch
batch_size = args.batch_size

# adjusting hidden layers if none is passed
if type(hidden_layers) == int:
    hidden_layers = [hidden_layers]

# loading all data loaders inside off a list
data_loaders, images_datasets = data_loader.load_images(data_dir=data_directory, batch_size=batch_size)

# building model
model = nn_model.build_model(arch=arch, hidden_layers=hidden_layers, dropout=dropout)

# check if the model returned a error
if model != 0:
    # checking if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # definening the optimizer to use only at train mode
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print('I\'m going to use the device: {}'.format(device))
    nn_model.run_train(model=model, data_loaders=data_loaders, optimizer=optimizer, epoch=epoch, device=device)

    print('creating checkpoint...')
    nn_model.save(model=model, arch=arch, dropout=dropout, learning_rate=learning_rate, image_datasets=images_datasets,
                  optimizer=optimizer, folder=save_directory, epoch=epoch)
