import torch
from torchvision import transforms, datasets


def load_images(data_dir='./flower', batch_size=32):
    '''
    This function will transforms the data set and return as dictionary with the data set
    Indexs:
    'train' - has the training data
    'valid' - validation data set
    'test' - test data set
    :param data_dir:
    :param batch_size:
    :return: dictionary with all data sets
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # to train -> Randomly rotate the images, resize, crop and flip the image
    data_transforms = transforms.Compose([transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # to test and valid -> resize and crop
    test_and_valid_transform = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms),
                      'valid': datasets.ImageFolder(valid_dir, transform=test_and_valid_transform),
                      'test': datasets.ImageFolder(test_dir, transform=test_and_valid_transform)}

    # Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
                    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True),
                    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)}

    return data_loaders, image_datasets
