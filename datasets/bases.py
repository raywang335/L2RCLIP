from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


vocab = {
    16:'sixteen',
    17:'Seventeen',
    18:'Eighteen',
    19:'Nineteen',
    20:'Twenty',
    21:'Twenty One',
    22:'Twenty Two',
    23:'Twenty Three',
    24:'Twenty Four',
    25:'Twenty Five',
    26:'Twenty Six',
    27:'Twenty Seven',
    28:'Twenty Eight',
    29:'Twenty Nine',
    30:'Thirty',
    31:'Thirty One',
    32:'Thirty Two',
    33:'Thirty Three',
    34:'Thirty Four',
    35:'Thirty Five',
    36:'Thirty Six',
    37:'Thirty Seven',
    38:'Thirty Eight',
    39:'Thirty Nine',
    40:'Forty',
    41:'Forty One',
    42:'Forty Two',
    43:'Forty Three',
    44:'Forty Four',
    45:'Forty Five',
    46:'Forty Six',
    47:'Forty Seven',
    48:'Forty Eight',
    49:'Forty Nine',
    50:'Fifty',
    51:'Fifty One',
    52:'Fifty Two',
    53:'Fifty Three',
    54:'Fifty Four',
    55:'Fifty Five',
    56:'Fifty Six',
    57:'Fifty Seven',
    58:'Fifty Eight',
    59:'Fifty Nine',
    60:'Sixty',
    61:'Sixty One',
    62:'Sixty Two',
    63:'Sixty Three',
    64:'Sixty Four',
    65:'Sixty Five',
    66:'Sixty Six',
    67:'Sixty Seven',
    68:'Sixty Eight',
    69:'Sixty Nine',
    70:'Seventy',
    71:'Seventy One',
    72:'Seventy Two',
    73:'Seventy Three',
    74:'Seventy Four',
    75:'Seventy Five',
    76:'Seventy Six',
    77:'Seventy Seven',
}

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        ages = []
        for _, age in data:
            ages += [age]
        ages = set(ages)
        num_ages = len(ages)
        num_imgs = len(data)
        return num_ages, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, test):
        num_train_ages, num_train_imgs = self.get_imagedata_info(train)
        num_test_ages, num_test_imgs = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ages | # images | ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | ".format(num_train_ages, num_train_imgs))
        print("  query    | {:5d} | {:8d} | ".format(num_test_ages, num_query_imgs))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pid2label=None):
        self.dataset = dataset
        self.transform = transform
        # self.vocab = {
        #     1:"one",
        #     2:"Two",
        #     3:"Three",
        #     4:"Four",
        #     5:"Five",
        #     6:"Six",
        #     7:"Seven",
        #     8:"Eight",
        #     9:"Nine",
        #     10:"Ten",
        #     11:"Eleven",
        #     12:"Twelve",
        #     13:"Thirteen",
        #     14:"Fourteen",
        #     15:"Fifteen",
        #     16:'sixteen',
        #     17:'Seventeen',
        #     18:'Eighteen',
        #     19:'Nineteen',
        #     20:'Twenty',
        #     21:'Twenty One',
        #     22:'Twenty Two',
        #     23:'Twenty Three',
        #     24:'Twenty Four',
        #     25:'Twenty Five',
        #     26:'Twenty Six',
        #     27:'Twenty Seven',
        #     28:'Twenty Eight',
        #     29:'Twenty Nine',
        #     30:'Thirty',
        #     31:'Thirty One',
        #     32:'Thirty Two',
        #     33:'Thirty Three',
        #     34:'Thirty Four',
        #     35:'Thirty Five',
        #     36:'Thirty Six',
        #     37:'Thirty Seven',
        #     38:'Thirty Eight',
        #     39:'Thirty Nine',
        #     40:'Forty',
        #     41:'Forty One',
        #     42:'Forty Two',
        #     43:'Forty Three',
        #     44:'Forty Four',
        #     45:'Forty Five',
        #     46:'Forty Six',
        #     47:'Forty Seven',
        #     48:'Forty Eight',
        #     49:'Forty Nine',
        #     50:'Fifty',
        #     51:'Fifty One',
        #     52:'Fifty Two',
        #     53:'Fifty Three',
        #     54:'Fifty Four',
        #     55:'Fifty Five',
        #     56:'Fifty Six',
        #     57:'Fifty Seven',
        #     58:'Fifty Eight',
        #     59:'Fifty Nine',
        #     60:'Sixty',
        #     61:'Sixty One',
        #     62:'Sixty Two',
        #     63:'Sixty Three',
        #     64:'Sixty Four',
        #     65:'Sixty Five',
        #     66:'Sixty Six',
        #     67:'Sixty Seven',
        #     68:'Sixty Eight',
        #     69:'Sixty Nine',
        #     70:'Seventy',
        #     71:'Seventy One',
        #     72:'Seventy Two',
        #     73:'Seventy Three',
        #     74:'Seventy Four',
        #     75:'Seventy Five',
        #     76:'Seventy Six',
        #     77:'Seventy Seven',
        # }
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, age = self.dataset[index]
        try:
            img = read_image(img_path)
        except:
            index = random.randint(0, len(self.dataset) - 1)
            img = read_image(self.dataset[index][0])
            print("read strange image:{}".format(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, age