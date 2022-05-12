import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def grid_gray_image(imgs, each_row: int):
    '''
    imgs shape: batch * size (e.g., 64x32x32, 64 is the number of the gray images, and (32, 32) is the size of each gray image)
    '''
    row_num = imgs.shape[0]//each_row
    for i in range(row_num):
        img = imgs[i*each_row]
        img = (img - img.min()) / (img.max() - img.min())
        for j in range(1, each_row):
            tmp_img = imgs[i*each_row+j]
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
            img = np.hstack((img, tmp_img))
        if i == 0:
            ans = img
        else:
            ans = np.vstack((ans, img))
    return ans

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps



# def featuremap_2_heatmap(feature_map):
#     assert isinstance(feature_map, torch.Tensor)
#     feature_map = feature_map.detach()
#     heatmap = feature_map[:,:]
#     heatmaps = []
#     # for c in range(feature_map.shape[1]):
#     #     heatmap+=feature_map[:,c,:,:]
#     heatmap = heatmap.cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
#
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     heatmaps.append(heatmap)
#
#     return heatmaps

# def featuremap_2_heatmap(feature_map):
#     assert isinstance(feature_map, torch.Tensor)
#     feature_map = feature_map.detach().squeeze(0).cpu()
#     ann = grid_gray_image(feature_map,16)
#     # heatmap = feature_map[:,0,:,:]*0
#     #
#     # heatmaps = []
#     # for c in range(feature_map.shape[1]):
#     #     # heatmap+=feature_map[:,c,:,:]
#     #     grid_gray_image
#     # heatmap = heatmap.cpu().numpy()
#     # heatmap = np.mean(heatmap, axis=0)
#     #
#     # heatmap = np.maximum(heatmap, 0)
#     # heatmap /= np.max(heatmap)
#     # heatmaps.append(heatmap)
#
#     return ann


def draw_feature_map(features,save_dir = '/home/yolov5_obb_bruce/visual_feature',name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        a = -1
        for i , featuremap in enumerate(features):

            heatmaps = featuremap_2_heatmap(featuremap.tensor)
            # if i in [0,1,3,4,6,7,9,10]:
            #     continue
            # a += 1
            # heatmaps = featuremap.numpy()
            for heatmap in heatmaps:
                # heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                plt.title(name[i%len(name)])
                plt.imshow(superimposed_img[...,::-1])
                plt.show()
                # plt.imsave()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir, 'rpn_cls'+ str(i) +'.jpg'), superimposed_img)
                # i=i+1
