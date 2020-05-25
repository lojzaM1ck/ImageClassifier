import argparse
import network_model

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='./flowers/test/1/image_06743.jpg', help='image to classify')
parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--top_k', default=5, type=int, help="Number of predicted classes")
parser.add_argument('--checkpoint',type=str, default='checkpoint.pth', help="path to the checkpoint")
parser.add_argument('--category_names', default = './cat_to_name.json', help="The file with category names")
parser.add_argument('--gpu', default=True, action='store_true', help="True = GPU False = CPU")


args = parser.parse_args()
img_path = args.img
data_directory = args.data_dir
checkpoint = args.checkpoint
category_name = args.category_names
top_k = args.top_k
gpu = args.gpu

model = network_model.load_checkpoint(checkpoint)
categories = network_model.load_mapping(category_name)
probs, classes = network_model.predict(img_path, model, gpu, top_k)

for (item, probability) in zip(classes, probs * 100):
    print('{} - {:.2f}%'.format(item, probability))
