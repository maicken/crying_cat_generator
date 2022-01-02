import keras, sys, cv2, os
from keras.models import load_model
import argparse
import numpy as np
import pandas as pd
from math import atan2, degrees, cos, sin, radians
import glob 


img_size = 224
crying_cat = cv2.imread('crying_cat_images/crying_cat_3.png', cv2.IMREAD_UNCHANGED)

left_eye = (270, 280)
right_eye = (675, 270)
size_between_eyes = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
center_cat = (498, 422)
adjust_angle = 0

# models
bbs_model_name = 'cat_hipsterizer/models/bbs_1.h5'
lmks_model_name = 'cat_hipsterizer/models/lmks_1.h5'
bbs_model = load_model(bbs_model_name)
lmks_model = load_model(lmks_model_name)

def resize_img(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=[0, 0, 0])
    return new_im, ratio, top, left

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, mask, x, y, overlay_size=None):
    bg_img = background_img.copy()

    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
        mask = cv2.resize(mask.copy(), overlay_size)
    
    h, w, _ = img_to_overlay_t.shape
    
    lim_left_x, lim_right_x, lim_up_y, lim_down_y = 0, bg_img.shape[1], 0, bg_img.shape[0]

    if int(y - h/2) < lim_up_y:
        margin =  lim_down_y - int(y - h/2)
        img_to_overlay_t = img_to_overlay_t[margin:, :]
        mask = mask[margin:, :]

    if int(y + h/2) > lim_down_y:
        margin = int(y + h/2) - lim_down_y
        img_to_overlay_t = img_to_overlay_t[:-margin, :]
        mask = mask[:-margin, :]

    if int(x - w/2) < lim_left_x:
        margin =  lim_left_x - int(x - w/2)
        img_to_overlay_t = img_to_overlay_t[:, margin:]
        mask = mask[:, margin:]

    if int(x + w/2) > lim_right_x:
        margin = int(x + w/2) - lim_right_x
        img_to_overlay_t = img_to_overlay_t[:, :-margin]
        mask = mask[:, :-margin]

    mask = cv2.medianBlur(mask, 5)
    mask1 = np.repeat(mask[:, :, np.newaxis], 4, axis=2) / 255
    mask2 = 1 - mask1

    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
    final = np.uint8(roi * mask2 + img_to_overlay_t * mask1)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = final

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))

def get_position(p, angle, distance):
    xDiff = distance * cos(radians(angle))
    yDiff = distance * sin(radians(angle))
    return (p[0] + int(xDiff), p[1] - int(yDiff))

def generate_crying_cat(f, output_path):
    img = cv2.imread(f)

    ori_img = img.copy()
    result_img = img.copy()

    # predict bounding box
    img, ratio, top, left = resize_img(img)

    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    # compute bounding box of original image
    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

    # compute lazy bounding box for detecting landmarks
    center = np.mean(ori_bb, axis=0)
    face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)

    # predict landmarks
    face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
    face_img, face_ratio, face_top, face_left = resize_img(face_img)

    face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

    pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

    # compute landmark of original image
    new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
    ori_lmks = new_lmks + new_bb[0]

    # visualize
    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

    for i, l in enumerate(ori_lmks):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # crying_cat
    crying_cat_width = crying_cat.shape[0]
    crying_cat_width = crying_cat_width * np.linalg.norm(ori_lmks[0] - ori_lmks[1]) / size_between_eyes

    angle = -angle_between(ori_lmks[0], ori_lmks[1])

    angle_left_center = -angle_between(left_eye, center_cat)
    distance_left_center = np.linalg.norm(np.array(left_eye) - np.array(center_cat))

    center_cat_rotated = get_position(tuple(ori_lmks[0]), angle + angle_left_center, 
                                        distance_left_center * crying_cat_width / crying_cat.shape[0])

    M = cv2.getRotationMatrix2D((crying_cat.shape[1] / 2, crying_cat.shape[0] / 2), angle + adjust_angle, 1)
    rotated_crying_cat = cv2.warpAffine(crying_cat, M, (crying_cat.shape[1], crying_cat.shape[0]))

    # mask
    mask = cv2.imread('crying_cat_images/mask.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    try:
        result_img = overlay_transparent(result_img, rotated_crying_cat, rotated_mask, center_cat_rotated[0], 
                                        center_cat_rotated[1], 
                                        overlay_size=(int(crying_cat_width), 
                                                    int(crying_cat.shape[0] * crying_cat_width / crying_cat.shape[1])))
    except Exception as e:
        print('failed overlay image')

    filename, ext = os.path.splitext(f)
    output_path = os.path.join(output_path, '%s_result%s' % (filename, ext))
    cv2.imwrite(output_path, result_img)


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-image', help='input image', type=str, metavar='<file>')
    parser.add_argument('-f', '--input-folder', help='input folder if you have multiple images', type=str, metavar='<path>')
    parser.add_argument('-o', '--output-path', help='output path', default='results', type=str, metavar='<path>')
    args = parser.parse_args()

    args = vars(parser.parse_args())

    if not args['input_image'] and not args['input_folder']:
        parser.error("must specify either -i or -f")
    
    if args['input_image']:
        generate_crying_cat(
            args['input_image'],
            args['output_path']
        )
    
    if args['input_folder']:
        for f in glob.glob(os.path.join(args['input_folder'], '*.jp*g')):
            generate_crying_cat(f,
                   args['output_path']
            )