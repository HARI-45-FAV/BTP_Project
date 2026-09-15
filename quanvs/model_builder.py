#model_builder.py
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
import torch.utils.data


class CNN(nn.Module):
    def __init__(self, device, config=None):
        super(CNN, self).__init__()
        # Unpack the configuration parameters for each layer

        if config is None:
            print("Using default configuration.")
            config = get_default_model_config()

        self.device = device
        self.config = config

        self.conv1 = nn.Conv2d(**config['conv1'])
        self.conv2 = None
        if 'conv2' in config:
            self.conv2 = nn.Conv2d(**config['conv2'])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout_conv = None
        self.dropout_fn = None

        if config['dropout_conv_rate'] > 0:
            self.dropout_conv = nn.Dropout(p=config['dropout_conv_rate'])
        if config['dropout_fc_rate'] > 0:
            self.dropout_fn = nn.Dropout(p=config['dropout_fc_rate'])

        fc_input_size = self.calculate_fc_input_size(config['input_shape'])
        self.fc1 = nn.Linear(fc_input_size, config['fc1']['out_features'])
        self.fc2 = nn.Linear(config['fc1']['out_features'], config['fc2']['out_features'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.dropout_conv is not None:
            x = self.dropout_conv(x)
        if self.conv2 is not None:
            x = self.pool(F.relu(self.conv2(x)))
            if self.dropout_conv is not None:
                x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        if self.dropout_fn is not None:
            x = self.dropout_fn(x)
        x = F.relu(self.fc1(x))
        if self.dropout_fn is not None:
            x = self.dropout_fn(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def calculate_fc_input_size(self, input_shape):
        # Temporarily forward pass through conv and pooling layers to find the input size to the first FC layer
        dummy_data = torch.zeros(1, *input_shape)  # Batch size of 1
        with torch.no_grad():
            dummy_data = self.pool(F.relu(self.conv1(dummy_data)))
            if self.conv2 is not None:
                dummy_data = self.pool(F.relu(self.conv2(dummy_data)))
        return int(torch.numel(dummy_data))
    
    def describe_architecture(self):
        print(self)
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("Number of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))


def get_default_model_config():
    return {
        'input_shape': (1, 30, 30),  # Example input shape (channels, height, width)
        'conv1': {'in_channels': 1, 'out_channels': 8, 'kernel_size': 3, 'padding': 0},
        'conv2': {'in_channels': 8, 'out_channels': 8, 'kernel_size': 3, 'padding': 0},
        'fc1': {'out_features': 32},
        'fc2': {'out_features': 7},
        'dropout_fc_rate': 0,
        'dropout_conv_rate': 0 
    }

def get_single_conv_model_config():
    return {
        'input_shape': (1, 30, 30),  # Example input shape (channels, height, width)
        'conv1': {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'padding': 0},
        'fc1': {'out_features': 32},
        'fc2': {'out_features': 7},
        'dropout_fc_rate': 0,
        'dropout_conv_rate': 0 
    }






def stack_quanv_on_top(quanv, model, verbose=False):

    class QuanvNN(nn.Module):
        def __init__(self, classical_component, new_layer):
            super(QuanvNN, self).__init__()
            self.quanv = new_layer
            self.classical_component = classical_component
            self.preprocessed = False
            
            first_layer = self.classical_component.conv1
            if first_layer is None:
                raise Exception("The model must have a Conv2d layer (conv1) as the first layer.")
            #check if the first layer is a convolutional layer
            if not isinstance(self.classical_component.conv1, nn.Conv2d):
                raise Exception("The first layer of the model must be a Conv2d layer.")
            
            new_input_channels = new_layer.out_channels

            kernel_size = first_layer.kernel_size[0]
            stride = first_layer.stride[0]
            padding = first_layer.padding
            input_channels = first_layer.in_channels
            if input_channels != 1:
                raise Exception("The input channels of the first layer must be 1.")
            output_channels = first_layer.out_channels

            self.classical_component.conv1 = nn.Conv2d(new_input_channels,
                                                  output_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding)
            
            self.device = self.classical_component.device
            
            if verbose:
                self.describe_architecture()

        def quanv_preprocess(self, dataloader, verbose=False):
            print("Preprocessing the dataset with the quanvolutional layer.")
            preprocessed_data = []
            targets = []
            batch_size = dataloader.batch_size
            for data, target in tqdm(dataloader):
                # data, target = data.to(self.device), target.to(self.device)
                output = self.quanv(data)
                preprocessed_data.append(output)
                if verbose:
                    quanv.print_counters()
                #print(output.shape) #ma dim  = 2, 5, 30, 30
                targets.append(target)
            preprocessed_data = torch.cat(preprocessed_data, dim=0)
            targets = torch.cat(targets, dim=0)
            preprocessed_dataset = torch.utils.data.TensorDataset(preprocessed_data, targets)
            preprocessed_loader = torch.utils.data.DataLoader(
                preprocessed_dataset,
                batch_size=batch_size,
                shuffle=True
                )
            return preprocessed_loader



        def multi_preprocess(self, dataloader):
            print("Preprocessing the dataset with the quanvolutional layer.")
            preprocessed_data = []
            targets = []
            batch_size = dataloader.batch_size

            def process_channel(data, i):
                return self.quanv.single_forward(data, i)

            for data, target in tqdm(dataloader):
                output_channels = self.quanv.out_channels
                futures = []
                with ThreadPoolExecutor(max_workers=300) as executor:
                    # Schedule the single_forward function to be executed for each channel
                    for i in range(output_channels):
                        futures.append(executor.submit(process_channel, data, i))

                    # Wait for all futures to complete and concatenate outputs
                    outputs = [future.result() for future in futures]
                    output = torch.cat(outputs, dim=1)

                preprocessed_data.append(output)
                targets.append(target)

            preprocessed_data = torch.cat(preprocessed_data, dim=0)
            targets = torch.cat(targets, dim=0)
            preprocessed_dataset = torch.utils.data.TensorDataset(preprocessed_data, targets)
            preprocessed_loader = torch.utils.data.DataLoader(
                preprocessed_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            return preprocessed_loader


        def forward(self, x):
            if not self.preprocessed:
                x = self.quanv(x)
            x = self.classical_component(x)
            return x
        
        def describe_architecture(self):
            print(self)
            try:
                print(self.quanv.encoding_approach, self.quanv.encoding_config)
            except:
                print("No encoding approach and configuration found.")
            #print("Number of parameters: ", sum(p.numel() for p in self.parameters()))
            print("Number of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    quanv_model = QuanvNN(model, quanv)
    return quanv_model
