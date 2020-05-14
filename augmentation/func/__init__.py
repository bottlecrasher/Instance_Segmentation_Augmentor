import numpy as np
import cv2
import random
from PIL import Image


class alpha_transformation:
    def __init__(self, transforms, autocrop=True, fill_RGBA=(0, 0, 0, 0)):
        self.transforms = transforms
        self.fill_RGBA = fill_RGBA
        self.autocrop = autocrop
        if hasattr(self.transforms, "fill"):
            self.transforms.fill = self.fill_RGBA
        elif hasattr(self.transforms, "fillcolor"):
            self.transforms.fillcolor = self.fill_RGBA

    def __str__(self):
        return str(self.transforms)

    def __call__(self, PIL_img):
        __ = self.transforms(PIL_img)
        return __.crop(__.getchannel(3).getbbox()) if self.autocrop else __

def biggest_contour(cnt):
    areas = [cv2.contourArea(c) for c in cnt]
    if areas ^ None
    max_index = np.argmax(areas)
    cnt = cnt[max_index]
    return np.array(cnt).reshape(cnt.shape[0], cnt.shape[2])


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
    return (x, y, w, h)


def enclosingcircle_radious(foreground, all_masked_pos_img):
    contours = cv2.findContours(np.array(foreground.getchannel(3), dtype=np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), radious = cv2.minEnclosingCircle(contours[1][0])
    x, y, radious = int(x), int(y), int(radious)
    return radious


def rectangle_diagonal(foreground, all_masked_pos_img):
    _, __, x, y = foreground.getbbox()
    return int(np.sqrt(x ** 2 + y ** 2))


def dilate_by_kernel(foreground, all_masked_pos_img, max_size=(512, 324)):
    __y, __x = all_masked_pos_img.shape
    ratio = max(max_size[0] / __y, max_size[1] / __x)
    __all_masked_pos_img = cv2.resize(all_masked_pos_img, (int(__y * ratio), int(__x * ratio)))
    __kernel_y, __kernel_x = np.array(foreground.size) * ratio
    __kernel_y, __kernel_x = int(__kernel_y), int(__kernel_x)
    __kernel = (np.array(foreground.getchannel(3).resize((__kernel_x, __kernel_y)).convert("1")) * 1).astype(np.uint8)
    return cv2.resize(cv2.dilate(all_masked_pos_img, __kernel, iterations=1), (__x, __y))


def clamp_coordinate(x, y, img):
    if type(img) == np.ndarray:
        return max(0, min(x, img.shape[1])), max(0, min(y, img.shape[0]))
    else:  # Suppose img is PIL.Image type object
        return max(0, min(x, img.size[0])), max(0, min(y, img.size[1]))


def bg_alpha_composite(foreground, all_masked_pos_img, overlap_func=dilate_by_kernel):
    """
    :param bg:
    :param foreground: PIL Image, RGBA format.
    :param dest:
    :param all_masked_pos_img: np.array
    :param overlap_func:
    :return:
    """
    __ = overlap_func(foreground, all_masked_pos_img)
    indices = np.indices(all_masked_pos_img.shape).transpose(1, 2, 0)
    foreground_alpha = np.array(foreground.getchannel(3), dtype=np.uint8)
    __ = indices[__ == 0]
    if len(__) == 0: return False, False, False, all_masked_pos_img
    __y, __x = __[random.randint(0, len(__))]  # center coordination. (y,x)
    __y, __x = __y - int(foreground_alpha.shape[0] / 2), __x - int(foreground_alpha.shape[1] / 2)  # desc
    #
    # convert to temporaly PIL format to handling alpha composite without offset calculation.
    bg_inter_x, bg_inter_y, bg_inter_w, bg_inter_h = intersection((__x, __y, foreground.size[0], foreground.size[1]),
                                                                  (0, 0, all_masked_pos_img.shape[1],
                                                                   all_masked_pos_img.shape[0]))
    fore_inter_x, fore_inter_y, fore_inter_w, fore_inter_h = bg_inter_x - __x, bg_inter_y - __y, bg_inter_w, bg_inter_h

    all_masked_pos_img[bg_inter_y:bg_inter_y + bg_inter_h, bg_inter_x:bg_inter_x + bg_inter_w] = \
        cv2.bitwise_or(all_masked_pos_img[bg_inter_y:bg_inter_y + bg_inter_h, bg_inter_x:bg_inter_x + bg_inter_w],
                       foreground_alpha[fore_inter_y:fore_inter_y + fore_inter_h,
                       fore_inter_x:fore_inter_x + fore_inter_w])
    # only extract
    segmentation, __2 = cv2.findContours(foreground_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = biggest_contour(segmentation)
    bbox = (bg_inter_x, bg_inter_y, bg_inter_w, bg_inter_h)
    foreground_cropped= foreground.crop((fore_inter_x, fore_inter_y, fore_inter_x+fore_inter_w, fore_inter_y+fore_inter_h))
    # coco format : BoxMode.XYWH_ABS, (x,y,w,h),
    return bbox, segmentation, foreground_cropped, all_masked_pos_img


