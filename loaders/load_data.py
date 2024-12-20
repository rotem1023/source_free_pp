import os
from loaders.data_list import ImageList_idx
from torch.utils.data import DataLoader
from torchvision import transforms

# class InputData:
    
#     def __init__(self, train_images, train_labels, validation_images, validation_labels):
#         self

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load_from_dir(dataset_dir): 
    train_bs = 32
    train_iamges = f"{dataset_dir}/train_images.txt"
    train_labels = f"{dataset_dir}/train_labels.txt"
    validation_iamges = f"{dataset_dir}/validation_images.txt"
    validation_labels = f"{dataset_dir}/train_labels.txt"
    
    
    txt_tar = open(train_iamges).readlines()
    labels_tar = open(train_labels).readlines()

    image_list = ImageList_idx(txt_tar, labels_tar, transform=image_test())
    train_loader = DataLoader(image_list, batch_size=train_bs, shuffle=False, 
        num_workers=2)
    
    txt_tar = open(validation_iamges).readlines()
    labels_tar = open(validation_labels).readlines()

    image_list = ImageList_idx(txt_tar, labels_tar, transform=image_test())
    val_loader = DataLoader(image_list, batch_size=train_bs, shuffle=False, 
        num_workers=2)


    return train_loader, val_loader


def _get_images_dir():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    return f"{current_file_path}/../data/images"


def _get_dataset_path(dataset, domain, year):
    dataset_dir = f"{_get_images_dir()}/{dataset}/{domain}/lists/{year}"
    return dataset_dir

def load_data(dataset, domain, year):
    dataset_dir = _get_dataset_path(dataset=dataset, domain=domain, year=year)
    return data_load_from_dir(dataset_dir=dataset_dir)