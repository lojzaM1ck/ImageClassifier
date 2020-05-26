import argparse
import network_model

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
parser.add_argument('--gpu', type=bool, default='True', help='True = gpu, False = cpu')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=1, help='num of epochs')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture: vgg16, densenet121, alexnet')
parser.add_argument('--batch_size', type=int, default=32,help='bacht size')
parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units for layer')

args = parser.parse_args()


data_directory = args.data_dir
save_directory = args.save_dir
arch = args.arch
learning_rate = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
batch_size = args.batch_size
gpu = args.gpu

dataloaders, image_datasets = network_model.preproces(data_directory)

model, criterion, optimizer  = network_model.create_model(arch, hidden_units, learning_rate, gpu)

trained_model=network_model.train_model(model, dataloaders, criterion, optimizer, epochs, gpu)

network_model.save_model(save_directory, image_datasets, arch, model, epochs, learning_rate, hidden_units, gpu)
