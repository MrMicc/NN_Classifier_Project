from typing import List, Any

import torch
from torch import nn, optim
from torchvision import models
from utils import my_classifier, image
import time
import matplotlib.pyplot as plt



import os


def build_model(arch='densenet121', hidden_layers=[256], dropout=0.005):
    '''
    This function will return a pre-trained model with a personalized classifier
    :return:
    :param arch: Pre-trained model arch, default is DENSENET121
    :param hidden_layers: list of hidden layers, default is [256]
    :param dropout: default is 0.005
    :return: personalized model if error 0 (zero) will be return
    '''

    print('building a pre-trained {} model '.format(arch))

    output_layer = 102
    if arch.lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_layer = 1024

    elif arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_layer = 25088
    else:
        print('Arch {} is not supported!!'.format(arch))
        return 0

    # freezing parameters to not backprogate though them
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier = my_classifier.Classifier(input_w=input_layer, output_w=output_layer, hidden_layers=hidden_layers,
                                                drop_out=dropout)

    print('Model was created with the following parameters:\n',
          'arch: {}\n'.format(arch),
          'input layer: {}\n'.format(input_layer),
          'output layer: {}\n'.format(output_layer),
          'hidden layers: {}\n'.format(hidden_layers),
          'dropout: {}\n'.format(dropout),
          '------------------------------------')
    return model


def run_train(model, data_loaders, optimizer, epoch=5, device='cpu'):
    start = time.time()


    # tracking the record to plot graph and to do further analize
    train_losses, valid_lossess, accuracy_train, accuracy_validation = [], [], [], []

    # Defining the Loss function
    criterion = nn.NLLLoss()

    epoch = epoch

    # moving model for correct device
    model.to(device)

    print('Just start the training witn {} epoch... This will take some time!\n'.format(epoch),
          'Relax and take a break and make a coffee!! ;)')

    for e in range(epoch):
        training_loss = 0
        training_accuracy = 0

        # making sure that model is in train modetraining_loss
        model.train()

        for images, labels in data_loaders['train']:

            # putting images and labels to correct device
            images, labels = images.to(device), labels.to(device)

            # reset the gradiants since the are cumulate
            optimizer.zero_grad()

            # getting the log probability
            lop_ps = model.forward(images)

            # getting the loss and than doing back propagation
            loss = criterion(lop_ps, labels)
            loss.backward()

            #optimizer the gradient
            optimizer.step()

            # adding up tranning loss
            training_loss += loss.item()

            # get class prediction
            ps = torch.exp(lop_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # accuracy prediction
            training_accuracy += torch.mean(equals.type(torch.FloatTensor))

        else:
            validation_loss = 0
            validation_accuracy = 0

            # setting up gradiants for validation off to save memory and computation
            with torch.no_grad():
                # settin the model to evaluation
                model.eval()

                for images, labels in data_loaders['valid']:
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model.forward(images)

                    loss = criterion(log_ps, labels)
                    validation_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_class, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)

                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor))

                # put to train again
                model.train()

        # Update the training and validation losses to graph the learning curve
        train_losses.append(training_loss / len(data_loaders['train']))
        valid_lossess.append(validation_loss / len(data_loaders['valid']))
        accuracy_train.append(training_accuracy / len(data_loaders['train']) * 100)
        accuracy_validation.append(validation_accuracy / len(data_loaders['valid']) * 100)

        # Print out the statistical information
        print('-----------------------\n',
              'Training Epoch: {}/{} ->'.format(e + 1, epoch),
              'TrainLoss: {:.3f} :'.format(training_loss / len(data_loaders['train'])),
              'Train Acc: {:.2f}% :'.format(training_accuracy / len(data_loaders['train']) * 100),
              'Valid Loss: {:.3f} :'.format(validation_loss / len(data_loaders['valid'])),
              'Valid Acc: {:.2f}% :'.format(validation_accuracy / len(data_loaders['valid']) * 100),
              'Time: {:.3f}'.format((time.time() - start)))


    #_plot_statics(data_a=train_losses, data_b=validation_loss, label_a='Train Loss', label_b='Validation Loss')


def save(model, arch, dropout, learning_rate, image_datasets, optimizer, epoch='', folder='./checkpoint'):
    # Good trick - put the model into CPU to avoid loading issues
    device = torch.device('cpu')
    model.to(device)
    # making sure it is not in train mode
    # preparing to save everything that is necessary
    checkpoint = {'arch': arch,
                  'input_w': model.classifier.input_w,
                  'output_w': model.classifier.output_w,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  # transform the hidden layers into an array
                  'drop_out': dropout,
                  'learn_rate': learning_rate,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': image_datasets['train'].class_to_idx
                  }

    # saving

    torch.save(checkpoint, os.path.join(folder, 'checkpoint-{}-{}.pth'.format(epoch, arch)))
    print('checkpoint saving ... checkpoint-{}-{}.pth'.format(epoch, arch))


def load(file_path):
    print('loading the last checkpoint')
    checkpoint = torch.load(file_path)
    model = models.densenet121(pretrained=True)
    model.classifier = my_classifier.Classifier(checkpoint['input_w'], checkpoint['output_w'], checkpoint['hidden_layers'],
                                                drop_out=checkpoint['drop_out'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('loading model {} with:\n'.format(checkpoint['arch']),
          'input: {}\n'.format(checkpoint['input_w']),
          'output: {}\n'.format(checkpoint['output_w']),
          'hidden layers: {}\n'.format(checkpoint['hidden_layers']),
          'dropout: {}\n'.format(checkpoint['drop_out']),
          'learn rate: {}\n'.format(checkpoint['learn_rate']))

    return model, optimizer


def _plot_statics(data_a, data_b, label_a, label_b):

    if os.environ['DISPLAY']:
        plt.plot(data_a, label=label_a)
        plt.plot(data_b, label=label_b)
        plt.legend(frameon=False)
        plt.show()
    else:
        print('This terminal has no Display, so I can\'t plot :(')


def predict(model: my_classifier.Classifier, image_path: str, topk=5):
    device = 'cpu'

    model.to(device)

    # making sure we are not in train mode
    model.eval()

    # we dont need to use gradients here, since we are not training
    with torch.no_grad():
        # getting image
        processed_img = image.process(image_path)

        processed_img = processed_img.type(torch.FloatTensor)

        # configuring image of flower

        processed_img.unsqueeze_(0)

        processed_img.to(device)

        # using model to make a prediction
        log_ps = model.forward(processed_img)
        ps = torch.exp(log_ps)

        # getting the 5 top probs and classes
        top_ps, top_classes = torch.topk(ps, topk)

    # very trick part - take hours to figure out
    # We need to reverse the catagories dictionary to make it work fine
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # get the labels
    labels = [idx_to_class[i.item()] for i in top_classes[0].data]

    return top_ps.numpy()[0].tolist(), labels


def santiy_check(model: my_classifier.Classifier, classes: str, file_path: str, img_index: str):
    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_axes([0, 0.55, .4, .4])
    result = process_image(file_path)
    ax = imshow(result, ax1)
    ax.axis('off')
    ax.set_title(cat_to_name[str(img_index)])

    # make a prediction based on an image
    predictions, prediciton_classes = predict(model=model, image_path=file_path)

    names = [classes[str(categorie)] for categorie in prediciton_classes]

    fig.add_axes([0, .1, .4, .4])
    print(predictions)
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, predictions, align='center')
    plt.yticks(y_pos, names)
    plt.xlabel('Probabilities')
    plt.show()
