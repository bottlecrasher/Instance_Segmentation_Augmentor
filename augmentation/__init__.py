import numpy as np
import random
from collections import Counter
import os
import cv2
from .func import alpha_transformation, bg_alpha_composite, biggest_contour
from PIL import Image
import json
from copy import deepcopy
from torchvision.datasets.folder import make_dataset, VisionDataset, IMG_EXTENSIONS
from torch.utils.data import DataLoader, SubsetRandomSampler


def rgba_pil_loader(path, mode="RGBA"):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def rgba_pil_fn(img):
    return img


class DatasetFolder_Subset(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        predefined_cls_to_idx (dictionary, optional): A dictionary that takes specific
            class name and its predefined index number. (used for loading subset of the dataset)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, predefined_cls_to_idx=None):
        super(DatasetFolder_Subset, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root, predefined_cls_to_idx)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir, predefined_cls_to_idx):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        if predefined_cls_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            classes = [cls for cls in classes if cls in predefined_cls_to_idx.keys()]
            class_to_idx = {cls_name: predefined_cls_to_idx[cls_name] for cls_name in classes
                            if cls_name in predefined_cls_to_idx.keys()}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageFolderSubset(DatasetFolder_Subset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=rgba_pil_loader, is_valid_file=None, predefined_cls_to_idx=None):
        super(ImageFolderSubset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                transform=transform,
                                                target_transform=target_transform,
                                                is_valid_file=is_valid_file,
                                                predefined_cls_to_idx=predefined_cls_to_idx)
        self.imgs = self.samples


class Foreground_Dataset():
    def __init__(self, root, transform=None, target_transform=None,
                 loader=rgba_pil_loader, is_valid_file=None, predefined_cls_to_idx=None):
        self.dataset = ImageFolderSubset(root, transform=transform, target_transform=target_transform,
                                         loader=loader, is_valid_file=is_valid_file,
                                         predefined_cls_to_idx=predefined_cls_to_idx)
        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes

        if getattr(self.dataset, "loader"):
            self.dataset.loader = rgba_pil_loader
        self.reload()

    def reload(self):
        self.__subset_iterators = dict.fromkeys([__cls_idx for __cls_idx in self.dataset.class_to_idx.values()])
        self.__imgpath_idxs_per_cls = deepcopy(self.__subset_iterators)

        for idx in sorted(self.__subset_iterators.keys()):
            self.__imgpath_idxs_per_cls[idx] = np.where(np.array(self.dataset.targets) == idx)[0]

        for idx in self.__subset_iterators.keys():
            self.__reset_iteration(idx)

    def __reset_iteration(self, idx):
        __ = np.array(deepcopy(self.__imgpath_idxs_per_cls[idx]))
        np.random.shuffle(__)
        self.__subset_iterators[idx] = iter(__)

    def __call__(self, idx):
        try:
            return self.dataset[next(self.__subset_iterators[idx])]
        except StopIteration:
            self.__reset_iteration(idx)
            return self.dataset[next(self.__subset_iterators[idx])]

    def __len__(self):
        return len(self.dataset)


class Instance_Segmentation_Augment_Dataset():
    def __init__(self, background_dataset,
                 foreground_dataset,
                 max_total=25,
                 max_iter=50,
                 strategy="fill",
                 sample_method="random"):
        """

        :param background_dataset:
        :param foreground_dataset:
        :param max_total:
        :param max_per_cls:
        :param max_iter:
        :param strategy: "fill", "flock",
        """
        self.bg_dataset = background_dataset
        self.fg_dataset = foreground_dataset
        self.fg_classes = self.fg_dataset.classes
        self.fg_class_to_idx = self.fg_dataset.class_to_idx
        self.fg_sample_method = sample_method
        self.strategy = strategy
        self.max_total = max_total
        self.max_iter = max_iter

    @staticmethod
    def __random_cls_sampler(classes, n):
        return random.choices(classes, k=n)

    def __getitem__(self, item):
        bg = self.bg_dataset[item][0]  # bg is PIL Image object.
        all_masked_pos_img = np.zeros(np.array(bg).shape[:2], dtype=np.uint8)
        foreground_list = []
        if self.fg_sample_method == "random":
            pull_list = self.__random_cls_sampler(self.fg_classes, self.max_total)

        for __cls in pull_list:
            cls_idx = self.fg_class_to_idx[__cls]
            bbox, segmentation, foreground, all_masked_pos_img = \
                self.__add_object(self.fg_dataset(cls_idx)[0], all_masked_pos_img)
            if bbox is False:
                continue
            new_foreground_attribute = (cls_idx, bbox, segmentation, foreground)
            foreground_list.append(new_foreground_attribute)
        return self.__flatten_objects_into_bg(bg, foreground_list)

    def __flatten_objects_into_bg(self, bg, foreground_list, order="random"):
        shuffle_idx = np.arange(len(foreground_list))
        segmentation_labels = np.ones((bg.size[1], bg.size[0]),
                                      dtype=np.uint8) * -1  # segmentation labels for shuffled index are stored in pixel-wise level
        cls_list = []
        bbox_list = []
        segmentation_list = []

        # Randomize Pasting order
        if order == "random":
            np.random.shuffle(shuffle_idx)

        # Pastes and generate new overlapping label map for randomized order
        for idx in shuffle_idx:
            cls_idx, bbox, segmentation, foreground = foreground_list[idx]
            foreground_alphamap = np.array(foreground.getchannel(3), dtype=np.uint8)
            x, y, w, h = bbox
            bg.alpha_composite(foreground, bbox[:2])
            segmentation_labels[y:y + h, x:x + w][foreground_alphamap != 0] = idx
            cls_list.append(cls_idx)

        for idx in shuffle_idx:
            __temp_segmentation_labels = deepcopy(segmentation_labels)
            __temp_segmentation_labels[__temp_segmentation_labels == idx] = 255
            __temp_segmentation_labels[__temp_segmentation_labels != 255] = 0
            __temp_segmentation_labels = __temp_segmentation_labels.astype(np.uint8)

            segmentation, __ = cv2.findContours(__temp_segmentation_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = biggest_contour(segmentation)
            new_x, new_y = segmentation.min(axis=0)
            new_xw, new_yh = segmentation.max(axis=0)
            new_bbox = (new_x, new_y, new_xw - new_x, new_yh - new_y)
            bbox_list.append(new_bbox)
            segmentation_list.append(segmentation)

        return cls_list, bbox_list, segmentation_list, bg

    def __add_object(self, object_img, all_masked_pos_img):
        bbox, segmentation, foreground, all_masked_pos_img = \
            bg_alpha_composite(object_img, all_masked_pos_img=all_masked_pos_img)
        return bbox, segmentation, foreground, all_masked_pos_img

    def __len__(self):
        return len(self.bg_dataset)