import torch
from torch import nn
import torch.nn.functional as F
import os.path
from helper import *


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = None, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        if len(hidden_layers) > 0: #hidden_layers
          # Input to a hidden layer
          self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
          
          # Add a variable number of more hidden layers
          layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
          self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
          
          self.output = nn.Linear(hidden_layers[-1], output_size)

        else:
          self.hidden_layers = None
          self.output = nn.Linear(input_size, output_size)
          
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        # if self.hidden_layers:
        if self.hidden_layers:
            for each in self.hidden_layers:
                x = F.relu(each(x))
                x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def load_model(params):

    arch = params['arch']
    model = params['model_dict'][arch](pretrained=True)
    for param in model.parameters():
      param.requires_grad = False

    input_num = params['fcs'][arch][1]
    hidden = params['hidden_units']
    print(hidden)
    drop_p = params['drop_p']

    c = Network(input_num, 102, hidden, drop_p)

    if hasattr(model, 'fc'):
        model.fc = c
    else:
        model.classifier = c

    check_path = params['check_dir'] + params['check_name'] + '.pth'

    if os.path.exists(check_path):
      print('exists')
      checkpt = torch.load(check_path)
      model.load_state_dict(checkpt['state_dict']) # was c before

    return model, c


def step(model, images, labels, device, trn = False):
      if trn:
        model.optimizer.zero_grad()

      images, labels = images.to(device), labels.to(device)
      log_ps = model(images)
      loss = model.criterion(log_ps, labels)
      
      if trn:
        loss.backward()
        model.optimizer.step()
      
      ps = torch.exp(log_ps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)

      accuracy = torch.mean(equals.type(torch.FloatTensor))
      # print(equals)
      print(f'Accuracy:{accuracy*100}% Loss: {loss.item()} -- {trn}')
      return loss.item(), accuracy.item()



def train(model, params, trainloader, testloader):

  save_file = f"{params['report_dir']}/{params['arch']}/{params['check_name']}"
  for epoch in range(params['epochs']):
      train_loss = 0
      test_loss = 0
      train_accuracy = 0
      test_accuracy = 0

      model.train()
      for images, labels in trainloader:
        loss, accuracy = step(model, images, labels, params['device'], True)
        train_loss += loss
        train_accuracy += accuracy
        
      save_checkpoint(params, model)

      model.eval()
      with torch.no_grad():
        for images, labels in testloader:
          loss, accuracy = step(model, images, labels, params['device'])
          test_loss += loss
          test_accuracy += accuracy
            
      print("Train Accuracy: {}:{}\n".format(epoch,str(train_accuracy/len(trainloader))))
      print("Test Accuracy {}:{}\n".format(epoch,str(test_accuracy/len(testloader))))
      print("Train Loss {}:{}\n".format(epoch,str(train_loss/len(trainloader))))
      print("Test Loss {}:{}\n".format(epoch,str(test_loss/len(testloader))))

def validate(model, params, validloader):
  save_file = f"{params['report_dir']}/{params['arch']}/{params['check_name']}"
  valid_loss = 0
  valid_accuracy = 0
  
  # model.eval()
  model.eval()
  with torch.no_grad():
    for images, labels in validloader:
          loss, accuracy = step(model, images, labels, params['device'])
          valid_loss += loss
          valid_accuracy += accuracy

  print(f"Final Validation Accuracy {100*valid_accuracy/len(validloader)}%\n%")
  print(f"Final Validation Loss {100*valid_loss/len(validloader)}%\n")

