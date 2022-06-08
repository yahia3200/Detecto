import cv2


def ImagesLoader(imgs_path):
    images = []
    for i in range(len(imgs_path)):
        images.append(cv2.imread(imgs_path[i]))

    return images  
