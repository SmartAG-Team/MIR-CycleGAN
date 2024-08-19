import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class MissdataDataset(BaseDataset):
    """A dataset class for temporal image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {{A_0, A_1},B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # 获得图片文件夹
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # 获取图片地址
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        # 裁剪的图片应该小于加载的图片的尺寸
        assert(self.opt.load_size >= self.opt.crop_size)
        # 输入通道数与输出通道数
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # 按照给定的随机索引顺序读取图片
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # AB = Image.open(AB_path)
        # self.input_nc = self.output_nc = self.opt.input_nc = self.opt.output_nc = 1
        # print("通道数：", self.opt.input_nc)

        # 分割拼接在一起的图片
        # A_miss：缺失数据影像
        # A_ref：参考影像
        # B_real：真实未缺失影像
        w, h = AB.size
        w4 = int(w / 3)
        A_miss = AB.crop((0, 0, w4, h))
        A_ref = AB.crop((w4, 0, 2*w4, h))
        B_real = AB.crop((2*w4, 0, w, h))
        # B_real = AB.crop((w4, 0, 2 * w4, h))
        # A_ref = AB.crop((2 * w4, 0, w, h))

        # apply the same transform to both A and B
        # 获取变换参数：裁剪位置、大小以及是否需要翻转
        transform_params = get_params(self.opt, A_miss.size)
        # 对对象进行上述变换，如果输入通道为1即转换为灰度图像
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A和B进行相同的变换以增加数据的多样性和一致性
        A_miss = A_transform(A_miss)
        A_ref = A_transform(A_ref)
        B_real = B_transform(B_real)

        return {'A_miss': A_miss, 'A_ref': A_ref, 'B_real': B_real, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
