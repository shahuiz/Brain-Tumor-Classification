import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from tqdm import tqdm

from PIL import Image
import torch
from torch.utils.data import Dataset

from resnet_model import resnet34

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    tumor_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = [] 
    val_images_path = []  
    val_images_label = [] 
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
   
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
   
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
   
        image_class = class_indices[cla]
   
        every_class_num.append(len(images))
   
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path: 
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else: 
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #train_path = "./data/Train/"
    #train_dataset = datasets.ImageFolder(root=train_path,                                     transform=data_transform["train"])
    #train_num = len(train_dataset)

    random.seed(0)
    images_path = "C:\\Users\\zhang\\Desktop\\BTC\\data\\Training"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(images_path)


    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    #flower_list = train_dataset.class_to_idx
    #tumour_state_list = train_dataset.class_to_idx
    #{'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
    #cla_dict = dict((val, key) for key, val in tumour_state_list.items())
    # write dict into json file
    #json_str = json.dumps(cla_dict, indent=4)
    #with open('class_indices.json', 'w') as json_file:
    #    json_file.write(json_str)

    batch_size = 16
    nw = 0
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "C:\\Users\\zhang\\Desktop\\BTC\\ResNet\\resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)


    #epochs = 3
    epochs = 1
    best_acc = 0.0
    save_path = './ResNet/models/resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
