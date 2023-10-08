from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os.path as osp
from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import os

random.seed(42)
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

class Morph(BaseImageDataset):
    """
    VehicleID
    Reference:
    Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles
    
    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    dataset_dir = 'Morph'

    def __init__(self, root='', train_txt_file='', test_txt_file='', verbose=True, test_size=800, idx=None, **kwargs):
        super(Morph, self).__init__()
        # self.dataset_dir = osp.join("/home/vcis11/userlist/share/Morph/image", self.dataset_dir)
        self.dataset_dir = root
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = train_txt_file
        self.test_size = 5021
        self.test_list = test_txt_file

        self.idx = idx
        self.check_before_run()

        train, test, val = self.process_split(relabel=True)
        self.train = train
        self.test = test
        self.val = val
        # self.train_pid2label = train_pid2label
        # self.gallery = gallery

        # if verbose:
        #     print('=> VehicleID loaded')
        #     self.print_dataset_statistics(train, test)

        self.num_train_ages, self.num_train_imgs = self.get_imagedata_info(
            self.train)
        self.num_test_ages, self.num_test_imgs = self.get_imagedata_info(
            self.test)
        self.num_val_ages, self.num_val_imgs = self.get_imagedata_info( self.val)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))


    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label


    def parse_img_ages(self, nl_pairs, pid2label=None, cam=0):
        output = []
        for info in nl_pairs:
            name = info[0]
            age = info[1]
            output.append((name, age))
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_age_dict = defaultdict(list)
        val_age_dict = defaultdict(list)

        with open(self.train_list) as f_train:
            train_data_list = list(f_train.readlines())
            train_ratio = 1.0
            random.shuffle(train_data_list)
            train_data = train_data_list[:int(len(train_data_list) * train_ratio)]
            # val_data = train_data_list[int(len(train_data_list) * train_ratio):]
            val_data = train_data
            # if self.idx is not None:
            #     fold_length = int(len(train_data_list) / 5)
            #     val_data = train_data_list[self.idx*fold_length:(self.idx+1)*fold_length]
            #     train_data = train_data_list[:self.idx*fold_length] + train_data_list[(self.idx+1)*fold_length:]
            # else:
            #     train_data = train_data_list[]
            #     val_data = train_data_list
            for data in train_data:
                name, age = data.strip().split(' ')
                age = int(age)
                if age > 77:
                    continue
                name = os.path.join(self.img_dir, os.path.basename(name))
                train_age_dict[age].append([name, age])
            for data in val_data:
                name, age = data.strip().split(' ')
                age = int(age)
                if age > 77:
                    continue
                name = os.path.join(self.img_dir, os.path.basename(name))
                val_age_dict[age].append([name, age])
        train_ages = list(train_age_dict.keys())
        val_ages = list(val_age_dict.keys())
        num_train_ages = len(train_age_dict)
        print("train ages:", num_train_ages, "val ages:", len(val_age_dict))

        test_age_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, age = data.strip().split(' ')
                age = int(age)
                if age > 77:
                    continue
                name = os.path.join(self.img_dir, os.path.basename(name))
                test_age_dict[age].append([name, age])
        test_ages = list(test_age_dict.keys())
        num_test_ages = len(test_age_dict)
        print("test ages:", num_test_ages)

        train_data = []
        val_data = []
        test_data = []
        train_ages = sorted(train_ages)
        val_ages = sorted(val_ages)
        test_ages = sorted(test_ages)
        # for train ids, all images are used in the train set.
        for age in train_age_dict:
            imginfo = train_age_dict[age]  # imginfo include image name and id
            train_data.extend(imginfo)
        
        for age in val_age_dict:
            imginfo = val_age_dict[age]  # imginfo include image name and id
            val_data.extend(imginfo)

        for age in test_age_dict:
            imginfo = test_age_dict[age]
            test_data.extend(imginfo)


        train = self.parse_img_ages(train_data)
        test = self.parse_img_ages(test_data)
        val = self.parse_img_ages(val_data)
        
        return train,test,val