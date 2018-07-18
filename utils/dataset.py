# Copyright      2018  Yiwen Shao

# Apache 2.0

import os
import torch
import time
import numpy as np
from tqdm import tqdm
import cv2
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


class AllDataset:
    """ It has class output and offset output as a combined target
    """

    def __init__(self, img_dir, annfile, num_classes, offset_list,
                 scale=1, crop=False, crop_size=None,
                 mode='train', limits=None, cache=False,
                 job=0, num_jobs=1):
        self.img_dir = img_dir
        self.coco = COCO(annfile)
        self.num_classes = num_classes
        self.offset_list = offset_list
        self.scale = scale
        self.crop = crop
        self.crop_size = crop_size
        if ((crop is False and crop_size is not None) or
                (crop is True and crop_size is None)):
            raise ValueError('crop and crop size should match')
        self.mode = mode
        if self.mode not in ['train', 'val', 'test', 'oracle']:
            raise ValueError('mode should be one of [train, val, test, oracle]'
                             'but given {}'.format(self.mode))
        # add a background class
        self.catIds = [0]
        self.catNms = ['background']
        self.ids = []
        self.ids = list(self.coco.imgs.keys())
        cats = self.coco.loadCats(self.coco.getCatIds())
        catNms_all = [cat['name'] for cat in cats]
        catIds_all = [cat['id'] for cat in cats]
        self.catIds.extend(catIds_all)
        self.catNms.extend(catNms_all)
        for i in range(len(self.catIds)):
            print('Class Name: {} \t Class Id:{} \t Category Id:{} \t'.format(
                self.catNms[i], i, self.catIds[i]))

        # if given, only the 'limits' number of samples in dataset is used
        if limits:
            self.limits = limits
            self.ids = self.ids[:limits]

        # for parallelization with multiple jobs (threads)
        self.job = job
        self.num_jobs = num_jobs
        assert job <= num_jobs
        if self.job > 0:  # job id is 1-indexed
            id_array = np.array(self.ids)
            self.ids = np.array_split(id_array, self.num_jobs)[
                self.job - 1].tolist()

        self.cache = cache
        if self.cache:
            start_time = time.time()
            self.all_imgs = []
            self.all_targets = []
            for img_id in tqdm(self.ids):
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
                self.all_imgs.append(img)
                self.all_targets.append(target)
            elapsed_time = time.time() - start_time
            print(
                "Finish pre-processing all images into target tensor"
                "and store them into memory, t={}s".format(elapsed_time))

    def __load_img(self, img_id):
        """ load image and its annotation.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds)
        anns = self.coco.loadAnns(ann_ids)
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.img_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = np.moveaxis(img, -1, 0)  # from (h, w, c) to (c, h, w)
        return img, anns

    def __prepare_image_and_target(self, img, anns):
        """ convert annotations to target
        """
        channel, height, width = img.shape
        mask, object_class = anns_to_mask(anns, height, width, self.catIds)
        # resize image and mask if needed
        if self.scale != 1:
            img, mask = resize_image_and_mask(img, mask, self.scale)
        # crop image and mask if needed
        if self.crop and not self.cache:  # do crop later if cache=True
            img, mask = crop_image_and_mask(img, mask, self.crop_size)

        target = self.__mask_to_target(mask, object_class)
        return img, target

    def __mask_to_target(self, mask, object_class):
        """ process mask and object_class into target
        """
        h, w = mask.shape
        target_dims = self.num_classes + len(self.offset_list)
        target = np.ndarray(shape=(target_dims, h, w), dtype='bool')

        # map object_id to class_id
        def obj_to_class(x):
            return object_class[x]

        class_mask = np.array([[obj_to_class(pixel)
                                for pixel in row] for row in mask])
        for n in range(self.num_classes):
            target[n, :, :] = (class_mask == n)

        for n, (i, j) in enumerate(self.offset_list):
            rolled_mask = np.roll(np.roll(mask, -i, axis=0), -j, axis=1)
            target[self.num_classes + n, :, :] = (rolled_mask == mask)

        return target

    def __to_tensor(self, img, target):
        img = torch.from_numpy(img.astype('float32') / 256)
        target = torch.from_numpy(target.astype('float32'))
        return img, target

    def __getitem__(self, index):
        img_id = self.ids[index]

        if self.mode == 'train':
            if self.cache:
                img = self.all_imgs[index]
                target = self.all_targets[index]
                if self.crop:
                    img, target = crop_image_and_target(
                        img, target, self.crop_size[0], self.crop_size[1])
            else:
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img, target

        elif self.mode == 'val':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img_id, img, target

        # return original image and it's size if it's for test
        elif self.mode == 'test':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img = img.astype('float32') / 256.0
            return img_id, img, (height, width)

        elif self.mode == 'oracle':
            img, anns = self.__load_img(img_id)
            ori_img = img
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)

            return img_id, ori_img, (height, width), target

    def __len__(self):
        return len(self.ids)


class OffsetDataset:
    """ It only has offset output as target
    """

    def __init__(self, img_dir, annfile, offset_list,
                 scale=1, crop_size=None, crop=False,
                 mode='train', limits=None, cache=False,
                 job=0, num_jobs=1):
        self.img_dir = img_dir
        self.coco = COCO(annfile)
        self.offset_list = offset_list
        self.scale = scale
        self.mode = mode
        if self.mode not in ['train', 'val', 'test', 'oracle']:
            raise ValueError('mode should be one of [train, val, test, oracle]'
                             'but given {}'.format(self.mode))
        self.crop = crop
        self.crop_size = crop_size
        if ((crop is False and crop_size is not None) or
                (crop is True and crop_size is None)):
            raise ValueError('crop and crop size should match')
        self.ids = []
        self.ids = list(self.coco.imgs.keys())

        # if given, only the 'limits' number of samples in dataset is used
        if limits:
            self.limits = limits
            self.ids = self.ids[:limits]

        # for parallelization with multiple jobs (threads)
        self.job = job
        self.num_jobs = num_jobs
        assert job <= num_jobs
        if self.job > 0:  # job id is 1-indexed
            id_array = np.array(self.ids)
            self.ids = np.array_split(id_array, self.num_jobs)[
                self.job - 1].tolist()

        self.cache = cache
        if self.cache:
            start_time = time.time()
            self.all_imgs = []
            self.all_targets = []
            for img_id in tqdm(self.ids):
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
                self.all_imgs.append(img)
                self.all_targets.append(target)
            elapsed_time = time.time() - start_time
            print(
                "Finish pre-processing all images into target tensor"
                "and store them into memory, t={}s".format(elapsed_time))

    def __load_img(self, img_id):
        """ load image and its annotation.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.img_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = np.moveaxis(img, -1, 0)  # from (h, w, c) to (c, h, w)
        return img, anns

    def __prepare_image_and_target(self, img, anns):
        """ convert annotations to target
        """
        channel, height, width = img.shape
        mask = anns_to_mask(anns, height, width)
        # resize image and mask if needed
        if self.scale != 1:
            img, mask = resize_image_and_mask(img, mask, self.scale)
        # crop image and mask if needed
        if self.crop and not self.cache:  # Note: we will do crop later if cache=True
            img, mask = crop_image_and_mask(
                img, mask, self.crop_size[0], self.crop_size[1])

        target = self.__mask_to_target(mask)
        return img, target

    def __mask_to_target(self, mask):
        """ process mask into target
        """
        h, w = mask.shape
        target_dims = len(self.offset_list)
        target = np.ndarray(shape=(target_dims, h, w), dtype='bool')

        for n, (i, j) in enumerate(self.offset_list):
            rolled_mask = np.roll(np.roll(mask, -i, axis=0), -j, axis=1)
            target[n, :, :] = (rolled_mask == mask)

        return target

    def __to_tensor(self, img, target):
        img = torch.from_numpy(img.astype('float32') / 256)
        target = torch.from_numpy(target.astype('float32'))
        return img, target

    def __getitem__(self, index):
        img_id = self.ids[index]

        if self.mode == 'train':
            if self.cache:
                img = self.all_imgs[index]
                target = self.all_targets[index]
                if self.crop:
                    img, target = crop_image_and_target(
                        img, target, self.crop_size, self.crop_size)
            else:
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img, target

        elif self.mode == 'val':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img_id, img, target

        # return original image and it's size
        elif self.mode == 'test':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img = img.astype('float32') / 256.0
            return img_id, img, (height, width)

        elif self.mode == 'oracle':
            img, anns = self.__load_img(img_id)
            ori_img = img
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)

            return img_id, ori_img, (height, width), target

    def __len__(self):
        return len(self.ids)


class ClassDataset:
    """ It only has class output as target
    """

    def __init__(self, img_dir, annfile,
                 scale=1, crop=False, crop_size=None, caffe=False,
                 mode='train', limits=None, cache=False,
                 job=0, num_jobs=1):
        self.img_dir = img_dir
        self.coco = COCO(annfile)
        self.scale = scale
        self.caffe = caffe
        self.mode = mode
        if self.mode not in ['train', 'val', 'test', 'oracle']:
            raise ValueError('mode should be one of [train, val, test, oracle]'
                             'but given {}'.format(self.mode))
        self.crop = crop
        self.crop_size = crop_size
        if ((crop is False and crop_size is not None) or
                (crop is True and crop_size is None)):
            raise ValueError('crop and crop size should match')
        self.ids = []
        self.ids = list(self.coco.imgs.keys())
        # add a background class
        self.catIds = [0]
        self.catNms = ['background']
        cats = self.coco.loadCats(self.coco.getCatIds())
        catNms_all = [cat['name'] for cat in cats]
        catIds_all = [cat['id'] for cat in cats]
        self.catIds.extend(catIds_all)
        self.catNms.extend(catNms_all)
        for i in range(len(self.catIds)):
            print('Class Name: {} \t Class Id:{} \t Category Id:{} \t'.format(
                self.catNms[i], i, self.catIds[i]))

        # if given, only the 'limits' number of samples in dataset is used
        if limits:
            self.limits = limits
            self.ids = self.ids[:limits]

        # for parallelization with multiple jobs (threads)
        self.job = job
        self.num_jobs = num_jobs
        assert job <= num_jobs
        if self.job > 0:  # job id is 1-indexed
            id_array = np.array(self.ids)
            self.ids = np.array_split(id_array, self.num_jobs)[
                self.job - 1].tolist()

        self.cache = cache
        if self.cache:
            start_time = time.time()
            self.all_imgs = []
            self.all_targets = []
            for img_id in tqdm(self.ids):
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
                self.all_imgs.append(img)
                self.all_targets.append(target)
            elapsed_time = time.time() - start_time
            print(
                "Finish pre-processing all images into target tensor"
                "and store them into memory, t={}s".format(elapsed_time))

    def __load_img(self, img_id):
        """ load image and its annotation.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.img_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = np.moveaxis(img, -1, 0)  # from (h, w, c) to (c, h, w)
        return img, anns

    def __prepare_image_and_target(self, img, anns):
        """ convert annotations to target
        """
        channel, height, width = img.shape
        # Note: it is instance-unawareness mask, just same as semantic segmentation
        mask = anns_to_mask_class(anns, height, width, self.catIds)
        # resize image and mask if needed
        if self.scale != 1:
            img, mask = resize_image_and_mask(img, mask, self.scale)
        # crop image and mask if needed
        if self.crop and not self.cache:  # Note: we will do crop later if cache=True
            img, mask = crop_image_and_mask(
                img, mask, self.crop_size[0], self.crop_size[1])

        target = self.__mask_to_target(mask)
        return img, target

    def __mask_to_target(self, mask):
        """ process mask into target
        """
        h, w = mask.shape
        target_dims = len(self.catIds)
        target = np.ndarray(shape=(target_dims, h, w), dtype='bool')

        for n in range(len(self.catIds)):
            target[n, :, :] = (mask == n)

        return target

    def __to_tensor(self, img, target):
        if not self.caffe:  # float in (0, 1), RGB
            img = torch.from_numpy(img.astype('float32') / 256)
        else:  # float range in (0, 256), normalized, and in BGR format
            img = img.astype('float32')
            img -= np.array([123.68, 116.779, 103.939])[:, None, None]
            img = np.copy(img[::-1, :, :])
            img = torch.from_numpy(img)

        target = torch.from_numpy(target.astype('float32'))
        return img, target

    def __getitem__(self, index):
        img_id = self.ids[index]

        if self.mode == 'train':
            if self.cache:
                img = self.all_imgs[index]
                target = self.all_targets[index]
                if self.crop:
                    img, target = crop_image_and_target(
                        img, target, self.crop_size[0], self.crop_size[1])
            else:
                img, anns = self.__load_img(img_id)
                img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img, target

        elif self.mode == 'val':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)
            return img_id, img, target

        # return original image and it's size
        elif self.mode == 'test':
            img, anns = self.__load_img(img_id)
            c, height, width = img.shape
            img = img.astype('float32') / 256.0
            return img_id, img, (height, width)

        elif self.mode == 'oracle':
            img, anns = self.__load_img(img_id)
            ori_img = img
            c, height, width = img.shape
            img, target = self.__prepare_image_and_target(img, anns)
            img, target = self.__to_tensor(img, target)

            return img_id, ori_img, (height, width), target

    def __len__(self):
        return len(self.ids)


def anns_to_mask(anns, height, width, catIds=None):
    """ Given the annotations (list of dicts), return its mask (instance-aware). 
        And if catIds is given, also return the object_class
    """
    mask = np.zeros((height, width), dtype='uint16')
    if catIds:
        object_class = [0]  # the background class id is 0
    object_id = 1  # start with 1
    for ann in anns:
        category_id = ann['category_id']
        rle = ann_to_rle(ann, height, width)
        # get binary mask for each object and multiple it by object id
        m = maskUtils.decode(rle) * (object_id)
        object_id += 1
        # merge it to a single mask. note: it won't overwrite overlapping part
        mask = m * (mask == 0) + mask
        if catIds:
            class_id = catIds.index(category_id)
            object_class.append(class_id)
    if catIds:
        return mask, object_class
    else:
        return mask


def anns_to_mask_class(anns, height, width, catIds):
    """ Given the annotations (list of dicts), return its mask (not instance-aware). 
    """
    mask = np.zeros((height, width), dtype='uint8')
    for ann in anns:
        category_id = ann['category_id']
        class_id = catIds.index(category_id)
        rle = ann_to_rle(ann, height, width)
        # get binary mask for each object and multiple it by class id
        m = maskUtils.decode(rle) * class_id
        mask = m * (mask == 0) + mask
    return mask


def ann_to_rle(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: RLE
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def resize_image_and_mask(img, mask, scale):
    """ Resize image and mask by a scale factor.
    """
    c, h, w = img.shape
    height = int(h / scale)
    width = int(w / scale)
    img = np.moveaxis(img, 0, -1)
    img = cv2.resize(img, (width, height))
    img = np.moveaxis(img, -1, 0)
    mask = cv2.resize(mask, (width, height),
                      interpolation=cv2.INTER_NEAREST)
    return img, mask


def crop_image_and_mask(img, mask, height, width):
    """
    Randomly crops image and mask. It zero-pads if the current
    image is smaller than that.
    """
    c, h, w = img.shape
    if h < height:
        diff = height - h
        top_pad = int(diff / 2)
        bot_pad = diff - top_pad
        img = np.pad(img, ((0, 0), (top_pad, bot_pad), (0, 0)), 'constant')
        mask = np.pad(mask, ((top_pad, bot_pad), (0, 0)), 'constant')
    if w < width:
        diff = width - w
        left_pad = int(diff / 2)
        right_pad = diff - left_pad
        img = np.pad(
            img, ((0, 0), (0, 0), (left_pad, right_pad)), 'constant')
        mask = np.pad(mask, ((0, 0), (left_pad, right_pad)), 'constant')

    c, h, w = img.shape
    top = np.random.randint(0, h - height + 1)
    left = np.random.randint(0, w - width + 1)
    img = img[:, top:top + height, left:left + width]
    mask = mask[top:top + height, left:left + width]

    return img, mask


def crop_image_and_target(img, target, height, width):
    """
    Randomly crops image and target. It zero-pads if the current
    image is smaller than that.
    """
    c, h, w = img.shape
    if h < height:
        diff = height - h
        top_pad = int(diff / 2)
        bot_pad = diff - top_pad
        img = np.pad(img, ((0, 0), (top_pad, bot_pad), (0, 0)), 'constant')
        target = np.pad(
            target, ((0, 0), (top_pad, bot_pad), (0, 0)), 'constant')
    if w < width:
        diff = width - w
        left_pad = int(diff / 2)
        right_pad = diff - left_pad
        img = np.pad(
            img, ((0, 0), (0, 0), (left_pad, right_pad)), 'constant')
        target = np.pad(
            target, ((0, 0), (0, 0), (left_pad, right_pad)), 'constant')

    c, h, w = img.shape
    top = np.random.randint(0, h - height + 1)
    left = np.random.randint(0, w - width + 1)
    img = img[:, top:top + height, left:left + width]
    target = target[:, top:top + height, left:left + width]

    return img, target


class COCOTestset:
    def __init__(self, img_dir, info_file, c_cfg, class_nms=None):
        self.img_dir = img_dir
        self.coco = COCO(info_file)
        self.c_cfg = c_cfg
        self.class_nms = class_nms  # if given, is a subset of all 81 categories
        # add a background class
        self.catIds = [0]
        if self.class_nms:
            cats = self.coco.loadCats(self.coco.getCatIds())
            all_class_nms = [cat['name'] for cat in cats]
            for class_nm in self.class_nms:
                if class_nm not in all_class_nms:
                    raise ValueError('the given class name {}'
                                     'should be included in the dataset'.format(class_nm))
            assert len(class_nms) + 1 == self.c_cfg.num_classes
            catIds = self.coco.getCatIds(catNms=self.class_nms)
            self.catIds.extend(catIds)
            self.ids = self.coco.getImgIds(catIds=catIds)
        else:
            self.ids = list(self.coco.imgs.keys())
            self.catIds.extend(self.coco.getCatIds())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.img_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_id

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    import torchvision
    import torch.utils.data
    img_dir = 'data/val'
    ann = 'data/annotations/instancesonly_filtered_gtFine_val.json'
    offset_list = [(-1, 0), (0, -1)]
    # trainset = OffsetDataset(img_dir, ann, offset_list)
    # trainset = ClassDataset(img_dir, ann, crop=True, crop_size=(384, 384))
    trainset = AllDataset(img_dir, ann, 9, offset_list)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1)
    img, target = next(iter(dataloader))
    torchvision.utils.save_image(img, 'raw.png')
    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            target[:, 9 + i:9 + i + 1, :, :], 'bound_{}.png'.format(i))
    for i in range(len(trainset.catIds)):
        torchvision.utils.save_image(
            target[:, i:i + 1, :, :], 'class_{}.png'.format(i))
