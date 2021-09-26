import cv2

def resize(img, scale):
    w, h = img.shape[0], img.shape[1]
    w_after = int(w*scale)
    h_after = int(h*scale)
    img_after = cv2.resize(img, (h_after, w_after), interpolation=cv2.INTER_AREA)
    return img_after

def resize_to_constant(img, shape):
    img_after = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
    return img_after