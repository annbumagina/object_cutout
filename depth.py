#import test_simple
#import depthestim
#import tensorflow1.predict
#import tensorflow1.models.fcrn
#import LFattNet_evalution
import cv2
import torch

import matplotlib.pyplot as plt

# class Struct:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
#
#
# def find_depth1():
#     args = {'model_name': 'mono+stereo_640x192', 'no_cuda': False, 'image_path': 'children.jpg'}
#     s = Struct(**args)
#
#     test_simple.test_simple(s)


# def find_depth2():
#    if __name__ == '__main__':
#        depthestim

# def find_depth3():
#     tensorflow1.predict.main()

def find_depth4():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
    img = cv2.imread("resized_img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # predict and resize back
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    #plt.imshow(output)
    #plt.show()
    return output