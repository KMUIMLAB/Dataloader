    import os
import glob
import csv
import numpy as np
import pandas as pd
import configparser
import cv2
import matplotlib.pyplot as plt
from Dataloader import CustomDataset




class VerifyTimestamp(CustomDataset):
    def __init__(self,video_data=video_data,HMI_data=HMI_data):
        super().init(self)
        self.video_data=video_data
        self.HMI_data=HMI_data
        # self.num=num
        self.HMI_time=(HMI_data['time'][num])

    def get_selfreporting(self):
        report=[]
        for x,y in zip(self.HMI_data['time'],self.HMI_data['status']):
            report.append([x,y])
        return report

    def get_image_list_fromtimestamp(self):
        image_list={}
        report=self.get_selfreporting()
        for id in range(len(self.video_data)):
            for x,y in report:
                    video=self.video_data[id].split('/')[-1]
                    if video.endswith("ir.png"):
                        continue
                    video_time=float(video[:-4])
                    if video_time>=(x-0.5) and video_time<=(x+0.5):
                        if x not in image_list:
                            image_list[x]=[self.video_data[id]]
                        else:
                            image_list[x].append(self.video_data[id])
        return image_list

    def __len__(self):
        return len(self.image_list)

    def choose_images(self, num):
        image_list=self.get_image_list_fromtimestamp()
        images=image_list[self.HMI_time]
        return images
    
    def show_images(self):
        images=self.choose_images()
        for i in range(len(images)):
            image=cv2.imread(images[i])
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image_index=i+1
            ttitle="Image{}".format(image_index)
            plt.subplot(1,len(images),image_index)
            plt.title(ttitle)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)
    
    def check_image_timestamp(self,n):
        images=self.choose_images()
        my_image=images[n]
        image_time=float(my_image.split('/')[-1][:-4])
        return image_time

    def verify_selfreporting(self,n):
        image_time=self.check_image_timestamp(n)
        result=abs(image_time-self.HMI_time)
        if result<0.1:
            return result,True
        else:
            return result,False

    

if __name__ == '__main__':
    check()
    data=CustomDataset(PATH='/home/imlab/Desktop/yjs/Data/26595', HMI='HMI',video='video',location='SIDE')
_, _, _, video_data, _, HMI_data, _=data.__getitem__()



