# Object cutout

Cuts out chosen object

## Usage

Run, click on the desired object and get cutout. Cutout and trimap will be saved in files image_cutout.png and image_trimap.png

Parameters:
- ```image_name``` path to your image
- ```margin``` if margin >= 1 then uses that margin otherwise it is a margin percent. Default 0.22
- ```segmentation``` segmentation algorithm: "semantic" or "instance". Default "instance"
- ```alpha_matting``` alpha matting algorithm: "cf", "rw", "knn", "lkm", "lbdm". Default "rw"

You can run it from command line with arguments in the specified order, or call ```find_and_cut``` function