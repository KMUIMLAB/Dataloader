import os
import glob
import csv
import numpy as np
import pandas as pd
import configparser
import cv2
from datetime import datetime
#from config import config

class CustomDataset():
    def __init__(self, path='26595', audio=False, can=False, gnss=False, video=False, bio=False, hmi=False, traffic_info=False, bio_list=['ACC','BVP','EDA','HR','IBI']):
        self.path = os.path.join("../Data",path)
        self.audio = audio
        self.can = can
        self.gnss = gnss
        self.video = video
        self.bio = bio
        self.hmi = hmi
        self.traffic_info = traffic_info
        self.bio_list = bio_list
        

    def __getitem__(self, idx):
        x = [x for x in range(1000)]

        return x[idx]
    
    def get_bio(bio_list):

        return 0
    
    def convert_utctime(self, time):
        year,month,day,hour,minute,second,mil_second=time.split('_')

        return datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),int(mil_second)).timestamp()

    def get_path(self, data, bio_list=False):
        if bio_list:
            path_list = self.get_bio(bio_list)
        else:
            path_list = glob.glob(os.path.join(self.path, data, '*.csv'))
        if len(path_list) > 1:
            raise ValueError("files are more than 1.")
        if data == 'video':
            video_center_img = glob.glob(os.path.join(self.path,data,'internal','CENTER','*'))
            video_side_img = glob.glob(os.path.join(self.path,data,'internal','SIDE','*'))
            video_center_img.sort()
            video_side_img.sort()
            return video_center_img, video_side_img
        df = self.get_csv(path_list[0])
        
        if data == 'GNSS':
            utc_time=[]
            for time in df['Timestamp']:
                utc_time.append(self.convert_utctime(time))
            df.insert(1, 'timestamp', utc_time)

        return df

    def get_csv(self, path):
        df = pd.read_csv(path)

        return df
    
    def return_(self):
        audio_data = self.get_path('audio') if self.audio else None
        can_data = self.get_path('CAN') if self.can else None
        gnss_data = self.get_path('GNSS') if self.gnss else None
        video_center_img, video_side_img = self.get_path('video') if self.video else None
        bio_data = self.get_path('bio', self.bio_list) if self.bio else None
        hmi_data = self.get_path('HMI') if self.hmi else None
        traffic_data = self.get_path('Traffic_info') if self.traffic_info else None

        return audio_data, can_data, gnss_data, video_center_img,video_side_img, bio_data, hmi_data, traffic_data
                

    def get_can_data(self, status, past_time, future_time):
        _, can_data, _, _, _, _, hmi_data, _ = self.return_()
        times = hmi_data[hmi_data['status']==status]['time']
        can_df = pd.DataFrame([])
        for time in times:
            start, end = time - past_time, time + future_time
            can_range = can_data[can_data['timestamp'].between(start,end)]
            can_df = can_df.append(can_range)

        return can_df
    
    def get_gnss_data(self, status, past_time, future_time):
        _, _, gnss_data, _, _, _, hmi_data, _ = self.return_()
        times = hmi_data[hmi_data['status']==status]['time']
        gnss_df = pd.DataFrame([])
        for time in times:
            start, end = time - past_time, time + future_time
            gnss_range = gnss_data[gnss_data['timestamp'].between(start,end)]
            gnss_df = gnss_df.append(gnss_range)

        return gnss_df
    
    def get_specific_data(self, df, data_name):

        return df.loc[:, data_name]


    def get_image_list(self, status, location, past_time, future_time):
        _, _, _, video_center_img,video_side_img, _, hmi_data, _= self.return_()
        times = hmi_data[hmi_data['status']==status]['time']
        if location not in ['CENTER', 'SIDE']:
            raise ValueError("location should be in 'CENTER' or 'SIDE'.")
        img_list = video_center_img if location == 'CENTER' else video_side_img
        i = 0
        result_img = []
        for time in times:
            start, end = time - past_time, time + future_time
            for id in range(i, len(img_list)):
                if img_list[id].endswith('ir.png'):
                    i+=1    
                    continue
                img_time = float(img_list[id].split('/')[-1][:-4])
                if img_time >= start and img_time <= end:
                    result_img.append(img_list[id])
                    i+=1
                if img_time >= end:
                    break

        return [result_img]
    
    def get_all_image_list(self, status, past_time, future_time):
        _, _, _, video_center_img,video_side_img, _, hmi_data, _= self.return_()
        times = hmi_data[hmi_data['status']==status]['time']
        center_id, side_id = 0, 0
        result_center_img, result_side_img = [], []
        for time in times:
            start, end = time - past_time, time + future_time
            for id in range(center_id, len(video_center_img)):
                if video_center_img[id].endswith('ir.png'):
                    center_id+=1    
                    continue
                center_img_time = float(video_center_img[id].split('/')[-1][:-4])
                if center_img_time >= start and center_img_time <= end:
                    result_center_img.append(video_center_img[id])
                    center_id+=1
                if center_img_time >= end:
                    break

            for id in range(side_id, len(video_side_img)):
                if video_side_img[id].endswith('ir.png'):
                    side_id+=1    
                    continue
                side_img_time = float(video_side_img[id].split('/')[-1][:-4])
                if side_img_time >= start and side_img_time <= end:
                    result_side_img.append(video_side_img[id])
                    side_id+=1
                if side_img_time >= end:
                    break

        return result_center_img, result_side_img

    def concat_imglist(self, img_list1, img_list2):
        img_list1=[img_list1]
        img_list1.append(img_list2)

        return img_list1
    
    def show_img_like_video(self, img, window_name, wait_time):
        new_img=cv2.imread(img,cv2.IMREAD_COLOR)
        cv2.imshow(window_name, new_img)
        cv2.waitKey(wait_time)

    def show_video(self, img_list):
        if len(img_list)==1:
            cv2.namedWindow('video',0)
            cv2.resizeWindow('video',640,480)
            cv2.moveWindow('video',40,30)
            for img in img_list:
                self.show_img_like_video(img, 'video', 66)
            cv2.destroyAllWindows()
            
            return 0

        elif len(img_list)==2:
            cv2.namedWindow('video1',0)
            cv2.resizeWindow('video1',640,480)
            cv2.moveWindow('video1',40,30)

            cv2.namedWindow('video2',0)
            cv2.resizeWindow('video2',640,480)
            cv2.moveWindow('video2',720,30)

            for img1,img2 in zip(img_list[0],img_list[1]):
                self.show_img_like_video(img1, 'video1', 33)
                self.show_img_like_video(img2, 'video2', 33)
            cv2.destroyAllWindows()
            
            return 0
        else:
            raise ValueError("img_list must be 1 or 2.")



    def save_img_to_video(self, img_list, video_name):
        video_path = os.path.join(self.path, video_name + '.mp4')
        fps = 15
        image = cv2.imread(img_list[0])
        height, width, _ = image.shape
        size = (width, height)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        for img in img_list:
            img=cv2.imread(img)
            out.write(img)
        out.release()

        
    
if __name__ == "__main__":
    Data=CustomDataset(path='26595', audio=False, can='CAN', gnss='GNSS', video='video', bio=False, hmi='HMI',traffic_info='Traffic_info',bio_list=['ACC','BVP','EDA'])
    audio_data, can_data, gnss_data, video_center_img,video_side_img, bio_data, hmi_data, traffic_data=Data.return_()
    center_img, side_img = Data.get_all_image_list(3,2,2)
    all_video = Data.concat_img(center_img, side_img)
    Data.show_video(all_video)

    