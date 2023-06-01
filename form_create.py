import cv2
import numpy as np 
from tqdm import tqdm
import random
import os
import imageio
import albumentations as A
from matplotlib import pyplot as plt
import shutil
from collections import defaultdict
import traceback



def fg_adjust(fg):
  alpha = random.uniform(0, 0.5)
  fg = cv2.convertScaleAbs(fg, alpha)
  # fg_hsv[:,:,0] += random.randrange(-100, 100)
  # fg_hsv[:,:,1] += random.randrange(-200, 200)
  
  return fg


def blend(bg, fg, x, y):
    """
        bg: background (color image)
        fg: foreground (color image)
        x, y: top-left point of foreground image (percentage)
    """
    x_root = x - 11
    y_root = y - 11
    # w = h = 25
    roi = bg[y_root:y_root+28, x_root:x_root+28]
    # h, w = bg.shape[:2]
    # x_abs, y_abs = int(x*w), int(y*h)
    fg = cv2.resize(fg, (28 ,28))
    new_fg = fg_adjust(fg)
    small_img_gray = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(small_img_gray, 50, 255, cv2.THRESH_BINARY)
    # cv2_imshow(roi)
    # ret, mask = cv2.threshold(fg, 120, 255, cv2.THRESH_BINARY)
    bg_mask = cv2.bitwise_or(roi,roi,mask = mask)
    # cv2_imshow(bg)
    mask_inv = cv2.bitwise_not(small_img_gray)
    fg = cv2.bitwise_and(fg,fg, mask=mask_inv)
    final_roi = cv2.add(bg_mask,new_fg)
    # print(final_roi.shape)
    bg[ y_root:y_root+28, x_root:x_root+28] = final_roi

    result = bg.copy()

    return result


#R 10 - 15 row 63 col 39
#1A 197 839 #1B 261 839 #31A 554 840 #61A 911 840 #91A 1267 840
#2A 197 878
#6A 197 1071
#11A 197 1303
#16A 197 1535
#21A 196 1767
#26A 196 1995


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def change_bg (image, bg):
    
    random_scale = random.randrange(2,4,1)
    
    offset = random.randrange(8,10,1)
        
    random_value = random.randrange(-100 , 100,10)
  
        
        
    image2add = cv2.resize(image, (bg.shape[0]//random_scale , bg.shape[1]//random_scale))
    image_bg = bg

    
    roi_bg = image_bg[bg.shape[0]//offset + random_value:bg.shape[0]//offset + random_value+image2add.shape[0],
                    bg.shape[1]//offset + random_value:bg.shape[1]//offset + random_value+image2add.shape[1]]


    image2add = np.where(image2add > 0, image2add, roi_bg) 
    image_bg[bg.shape[0]//offset + random_value:bg.shape[0]//offset + random_value+image2add.shape[0],
            bg.shape[1]//offset + random_value:bg.shape[1]//offset + random_value+image2add.shape[1]] = image2add

    return image_bg
        


def gen_form (form_path, num_gen, save_path, bg_path = None):
    
    path = "/home/pvt/formcv/marks"
    mark_samples = os.listdir(path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    y_root_coor = [839, 1071, 1303, 1535, 1767, 1995]
    x_root_coor = [197, 554, 911, 1265]
    # img = cv2.imread ("/home/pvt/formcv/Mau_Phieu_120cau_BGD.png")

    if bg_path == None:
        for count_img in tqdm(range(num_gen)):
            img2overlay = cv2.imread ("/home/pvt/formcv/Mau_Phieu_120cau_BGD.png")
            for n, x in enumerate (x_root_coor):
                for m,y in enumerate( y_root_coor):
                
                    if m > 3:
                        x = x - 1
                    if n > 1:
                        y = y + n 
                    for i in range(5):
                        
                        tmp_y = y  + i *39
                        tmp_x = x 
                        ans = random.choice([0, 1, 2 ,3])
                        tmp_x += ans*63
                        overlay = cv2.imread(os.path.join(path, random.choice(mark_samples)))
                        res = blend(img2overlay, overlay, tmp_x, tmp_y)

            file_name = save_path + "/syn"+str(count_img)+".jpg"
#             print(i)
            filter_res = np.where(res < 100, 1, res)
            res = res + filter_res
            cv2.imwrite(file_name, res)

    else:
        bg_list = os.listdir(bg_path)
    
        for count_img in tqdm(range(num_gen)):
            bg = cv2.imread(os.path.join("/home/pvt/formcv/background", random.choice(bg_list)))
            img2overlay = cv2.imread ("/home/pvt/formcv/Mau_Phieu_120cau_BGD.png")

            for n, x in enumerate (x_root_coor):
                for m,y in enumerate( y_root_coor):
                
                    if m > 3:
                        x = x - 1
                    if n > 1:
                        y = y + n 
                    for i in range(5):
                        
                        tmp_y = y  + i *39
                        tmp_x = x 
                        ans = random.choice([0, 1, 2 ,3])
                        tmp_x += ans*63
                        overlay = cv2.imread(os.path.join(path, random.choice(mark_samples)))
                        res = blend(img2overlay, overlay, tmp_x, tmp_y)

            file_name = save_path + "/syn"+str(count_img)+".jpg"
            # print(i)
            filter_res = np.where(res < 100, 1, res)
            res = res + filter_res
            rotated = rotate_image(res, random.randrange(0, 15, 1))
            changed = change_bg(rotated, bg)
            # print(changed)
            cv2.imwrite(file_name, changed)

if __name__ == "__main__":


    gen_form(form_path="/home/pvt/formcv/Mau_Phieu_120cau_BGD.png", save_path= "/home/pvt/formcv/formtest/", num_gen=50, bg_path= "/home/pvt/formcv/background")


