import os
import shutil


if __name__ == '__main__':
    path_images =r'/home/DroneVehicleDataNoEdgeAll/val/images'
    path_labels = r'/home/DroneVehicleDataNoEdgeAll/train/labels'
    save_images = r'/home/DroneVehicleDataNoEdgeAll/train/no_rgb_images'
    save_labels = r'/home/DroneVehicleDataNoEdgeAll/train/no_rgb_labels'
    images = os.listdir(path_images)
    with open(r'/home/DroneVehicleDataNoEdgeAll/val/imgnamefile.txt','w',encoding='utf-8') as f:
        for image in images:
            f.write(image.replace('.jpg','\n').replace('.JPG','\n'))

            # if 'no_rgb' in image:
            #     shutil.move(os.path.join(path_images,image),save_images)
            #     shutil.move(os.path.join(path_labels,image.replace('.jpg','.txt')),save_labels)




