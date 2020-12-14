from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from pymatting import *
import sys
from math import ceil, floor, sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import copy
import cv2
import scipy.misc
import random

import depth

steps = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
click = (-1, -1)


def shrink_segmap(image, margin):
    if margin == 0:
        return
    n, m = image.shape
    que = list()
    for i in range(n):
        for j in range(m):
            if image[i, j] > 0:
                if i == 0 or j == 0 or i == n - 1 or j == m - 1:
                    que.append((i, j))
                    image[i, j] = -1
                else:
                    for step_i, step_j in steps:
                        if image[i + step_i, j + step_j] == 0 and image[i, j] > 0:
                            que.append((i, j))
                            image[i, j] = -1
    for k in range(margin - 1):
        new_que = list()
        for i, j in que:
            for step_i, step_j in steps:
                new_i = i + step_i
                new_j = j + step_j
                if 0 < new_i < n and 0 < new_j < m \
                        and image[new_i, new_j] > 0:
                    new_que.append((new_i, new_j))
                    image[new_i, new_j] = -1
        que = new_que


def expand_segmap(image, margin):
    if margin == 0:
        return
    n, m = image.shape
    que = list()
    for i in range(n):
        for j in range(m):
            if image[i, j] == 0:
                for step_i, step_j in steps:
                    new_i = i + step_i
                    new_j = j + step_j
                    if 0 < new_i < n and 0 < new_j < m \
                            and image[new_i, new_j] == -1 and image[i, j] == 0:
                        image[i, j] = -2
                        que.append((i, j))
    for k in range(margin - 1):
        new_que = list()
        for i, j in que:
            for step_i, step_j in steps:
                new_i = i + step_i
                new_j = j + step_j
                if 0 < new_i < n and 0 < new_j < m \
                        and image[new_i, new_j] == 0:
                    new_que.append((new_i, new_j))
                    image[new_i, new_j] = -2
        que = new_que


def to_trimap(image):
    n, m = image.shape
    trimap = np.empty((n, m), 'float64')
    for i in range(n):
        for j in range(m):
            if image[i, j] < 0:
                trimap[i, j] = 0.5
            elif image[i, j] > 0:
                trimap[i, j] = 1
            else:
                trimap[i, j] = 0
    return trimap


def onclick(event):
    global click
    if event.xdata is not None:
        click = (event.xdata, event.ydata)
        plt.close()


def find_box(img_size, x, y):
    width, height = img_size
    a = 0
    b = 0
    c = width
    d = height
    if width < height:
        if b + (width + 1) / 2 > x:
            d = width
        elif d - (width + 1) / 2 < x:
            b = d - width
        else:
            b = x - floor(width / 2.0)
            d = x + ceil(width / 2.0)
        x -= b
    elif height < width:
        if a + (height + 1) / 2 > y:
            c = height
        elif c - (height + 1) / 2 < y:
            a = c - height
        else:
            a = y - floor(height / 2.0)
            c = y + ceil(height / 2.0)
        y -= a
    return x, y, (a, b, c, d)


def get_margin(margin, om):
    if margin < 1:
        return round(sqrt(100 * margin * np.count_nonzero(om)) / 100)
    return int(margin)


def semantic_segmentation(img):
    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    y = int(round(click[0]))
    x = int(round(click[1]))

    min_side = min(img.size)
    if x != -1:
        x, y, box = find_box(img.size, x, y)
        img = img.crop(box)
    else:
        img = T.CenterCrop(min_side).forward(img)
    img.save("resized_img.jpg")
    img = T.Resize(224).forward(img)
    img = T.CenterCrop(224).forward(img)
    trf = T.Compose([T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)

    # pass image through network
    out = fcn(inp)['out']
    out = T.Resize(min_side).forward(out)
    # get class for each pixel
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    if x != -1 and om[x, y] > 0:
        cl = om[x, y]
        om = (om == cl).astype(int)
    else:
        om = (om > 0).astype(int)
    return om


def instance_segmentation(img, margin):
    y = int(round(click[0]))
    x = int(round(click[1]))
    print(x, y)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > 0.5][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]

    if (pred_t == 0):
        raise Exception("No objects were found")
    pos = 0
    print(masks.shape)
    if x != -1:
        for i in range(len(masks)):
            if masks[i][x, y]:
                pos = i

    ((x1, y1), (x2, y2)) = pred_boxes[pos]
    om = masks[pos].astype(float)
    margin = get_margin(margin, om)
    x1 = floor(x1) - margin
    y1 = floor(y1) - margin
    x2 = ceil(x2) + margin
    y2 = ceil(y2) + margin
    img = np.stack(img.numpy(), axis=2)
    img = img[y1:y2 + 1, x1:x2 + 1]
    save_image("resized_img.jpg", img)

    return om[y1:y2 + 1, x1:x2 + 1]


def color_diff(color1, color2):
    color1_rgb = sRGBColor(*color1)
    color2_rgb = sRGBColor(*color2)
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e


def clean_up_alpha(alpha, image, margin, color_dist=2, taken_value=0.8, neighbors=1):
    print(alpha.shape)
    # alpha[alpha < 0.4] = 0
    plt.imshow(alpha)
    plt.show()
    n = image.shape[0]
    m = image.shape[1]
    distance = np.zeros((n, m))
    # 0 - not looked, 1 - processing/bad, 2 - good
    good = np.zeros((n, m))
    good[alpha >= taken_value] = 2
    que = list()
    for i in range(n):
        for j in range(m):
            if alpha[i, j] < taken_value:
                for step_i, step_j in steps:
                    new_i = i + step_i
                    new_j = j + step_j
                    if 0 < new_i < n and 0 < new_j < m \
                            and alpha[new_i, new_j] >= taken_value:
                        que.append((i, j))
                        good[i, j] = 1
                        distance[i, j] = 1
                        break
    c = 0
    while True:
        # plt.imshow(good)
        # plt.show()
        c += 1
        new_que = []
        # image[i, j] is not taken yet
        for i, j in que:
            neis = 0
            dist = 2 * margin
            for step_i, step_j in steps:
                new_i = i + step_i
                new_j = j + step_j
                if 0 < new_i < n and 0 < new_j < m and good[new_i, new_j] == 2 \
                        and color_diff(image[i, j], image[new_i, new_j]) < color_dist:
                    neis += 1
                    dist = min(dist, distance[new_i, new_j])
            if neis >= neighbors and dist < margin:
                good[i, j] = 2
                distance[i, j] = dist + 1
                for step_i, step_j in steps:
                    new_i = i + step_i
                    new_j = j + step_j
                    if 0 < new_i < n and 0 < new_j < m and good[new_i, new_j] == 0 \
                            and alpha[new_i, new_j] > 0.01:
                        good[new_i, new_j] = 1
                        new_que.append((new_i, new_j))
        que = new_que
        if len(que) == 0:
            break
    plt.imshow(good)
    plt.show()
    alpha[good < 2] = 0
    print(c)
    # alpha[good == True] = 1


def cutout_object(image_path, trimap_path, margin, alpha_matting):
    scale = 1.0
    image = load_image(image_path, "RGB", scale, "box")
    trimap = load_image(trimap_path, "GRAY", scale, "nearest")
    # estimate alpha from image and trimap
    if alpha_matting == "cf":
        alpha = estimate_alpha_cf(image, trimap)
    elif alpha_matting == "rw":
        alpha = estimate_alpha_rw(image, trimap)
    elif alpha_matting == "knn":
        alpha = estimate_alpha_knn(image, trimap)
    elif alpha_matting == "lkm":
        alpha = estimate_alpha_lkm(image, trimap)
    elif alpha_matting == "lbdm":
        alpha = estimate_alpha_lbdm(image, trimap)
    else:
        raise Exception("No such alpha matting algorithm")
    # estimate foreground from image and alpha
    beta = copy.deepcopy(alpha)
    beta[beta < 0.6] = 0
    foreground = estimate_foreground_ml(image, alpha)
    # save cutout
    cutout = stack_images(foreground, alpha)
    save_image("image_cutout.png", cutout)

    clean_up_alpha(alpha, image, margin)
    foreground = estimate_foreground_ml(image, alpha)
    # save cutout
    cutout = stack_images(foreground, alpha)
    save_image("image_cutout2.png", cutout)

    foreground = estimate_foreground_ml(image, beta)
    # save cutout
    cutout = stack_images(foreground, beta)
    save_image("image_cutout3.png", cutout)


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def find_and_cut(image_name, margin=0.22, segmentation="instance", alpha_matting="cf"):
    margin = float(margin)
    img = Image.open(image_name).convert('RGB')
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click on the desired object.\nYou have only one try, so choose wisely!")
    plt.show()

    if segmentation == "semantic":
        om = semantic_segmentation(img)
    elif segmentation == "instance":
        om = instance_segmentation(img, margin)
    else:
        raise Exception("No such segmentation algorithm")

    # shows segmentation
    img = cv2.imread("resized_img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_mask = random_colour_masks(om)
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    cv2.imwrite("image_weighted.jpg", img)
    plt.imshow(img)
    plt.show()

    # calculates trimap
    ####################################
    img_depth = depth.find_depth4()
    mmean = img_depth.mean()
    height = len(img_depth)
    img_depth[3 * height // 4:] = mmean - 1
    om[img_depth >= mmean] = 1
    # for i in range(om.shape[0]):
    #     for j in range(om.shape[1]):
    #         if (om[i, j] < 0.9 and img_depth[i, j] > mmean):
    #             om[i, j] = 0.5
    ####################################
    margin = get_margin(margin, om)
    shrink_segmap(om, margin)
    expand_segmap(om, margin)
    trimap = to_trimap(om)
    plt.imshow(trimap)
    plt.title("Here's your trimap")
    plt.show()
    save_image("image_trimap.png", trimap)

    # cutout object
    cutout_object("resized_img.jpg", "image_trimap.png", margin, alpha_matting)
    img_cutout = Image.open("image_cutout.png")
    plt.imshow(img_cutout)
    plt.title("and cutout")
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    find_and_cut("girls.jpg", margin=0.1)
    img_array = depth.find_depth4()
    # img_array = np.load('girl_one.npy')
    mmean = img_array.mean()
    height = len(img_array)
    img_array[3 * height // 4:] = mmean - 1
    img_array[img_array < mmean] = mmean
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    print(np.max(img_array))
    print(np.mean(img_array))
    plt.imshow(img_array * 255)
    plt.show()
    im = Image.fromarray(img_array * 255)
    im = im.convert("RGB")
    im.save("image_depth.png")
