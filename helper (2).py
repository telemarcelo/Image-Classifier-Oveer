import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os.path, os


def set_params(train_params): 
  if torch.cuda.is_available() and train_params['gpu']:
    device = torch.device('cuda') 
  else:
    device = torch.device('cpu')
    train_params['gpu'] = False

  train_params['device'] = device

  # train_params['check_name']
  train_params['check_name'] = ''
  for prop in ['arch', 'hidden_units','learning_rate', 'epochs', 'gpu', 'drop_p']:
    train_params['check_name'] += str(train_params.get(prop, '')) + "_"
    
  if train_params['hidden_units'] != '[]':
    train_params['hidden_units'] =  [int(item) for item in train_params['hidden_units'][1:-1].split(',')]
  else:
    train_params['hidden_units'] = []

  train_params['report_dir'] = train_params['save_dir'] 
  train_params['check_dir'] = train_params['save_dir']

  train_params['model_dict'] = {'resnet18': models.resnet18, 'resnet50' : models.resnet50, 'alexnet': models.alexnet, 
            'vgg16': models.vgg16, 'densenet121' : models.densenet121}

  train_params['fcs'] = {'resnet18': ['fc',512], 'resnet50' : ['fc',2048],  'alexnet': ['classifier', 9216],
        'vgg16': ['classifier', 25088], 'densenet121' : ['classifier',1024]}

def get_data(data_dir):
    # data_dir = args.data_dir

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    data_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, data_transforms)

    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
    
    return trainloaders, testloaders, validloaders, train_datasets.class_to_idx


def save_checkpoint(params, model):

  checkpt = {'input_size': params['fcs'][params['arch']][1],
              'output_size': 102,
              'hidden_layers': params['hidden_units'],
              'state_dict': model.state_dict(),
              'class_to_idx': params['class_to_idx']} # was m before

  torch.save(checkpt, params['save_dir'] + params['check_name'] + '.pth')

import signal

from contextlib import contextmanager

import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler

@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    print(1)
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    print(1)
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str)
    parser.add_argument('--save_dir', type = str, default = '')        
    parser.add_argument('--arch', type = str, default = 'vgg16')        
    parser.add_argument('--learning_rate', type = float, default = 0.01)            
    parser.add_argument('--hidden_units', type = str, default = '[512]')           
    parser.add_argument('--epochs', type = int, default = 20)   
    parser.add_argument('--gpu', type = bool, default = True)  
    parser.add_argument('--drop_p', type = float, default = 0.3)               
    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser()
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
    parser.add_argument('--data_dir', type = str)
    parser.add_argument('--save_dir', type = str, default = '')        
    parser.add_argument('--arch', type = str, default = 'vgg16')        
    parser.add_argument('--learning_rate', type = float, default = 0.01)            
    parser.add_argument('--hidden_units', type = str, default = '[512]')           
    parser.add_argument('--epochs', type = int, default = 20)   

    parser.add_argument('--drop_p', type = float, default = 0.3)   
    parser.add_argument('image_path', type = str)
    parser.add_argument('checkpoint', type = str)        
    parser.add_argument('--top_k', type = int, default = 5)        
    parser.add_argument('--category_names', type = str, default = "cat_to_name.json")               
    parser.add_argument('--gpu', type = bool, default = True)             
    return parser.parse_args()





    