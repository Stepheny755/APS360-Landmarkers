from dotenv import dotenv_values

import multiprocessing
import pandas as pd
import numpy as np
import shutil
import csv
import os


class Data_Splitting():

    def __init__(self,dataset_path:str, output_path:str, num_workers:int=8):
        print("Reading dataset from \"{}\" using {} workers".format(dataset_path,num_workers))
        print("New datasets will go into \"{}\"".format(output_path))

        self.data_path = dataset_path
        self.out_path = output_path
        self.num_workers = num_workers

        self.df = self.load_csv()
        self.classes = self.create_dict(self.df)
        self.num_classes = len(self.classes.items())

        # Enable multiprocessing
        self.mp = False



        # print((self.classes[1][0]["id"]))

        # print((self.classes[1][2]["id"]))
        # print(self.get_lm_from_id(1,2))
        # print(type(self.classes))
        # print(self.resolve_img_path('17660ef415d37059'))
        # print(self.resolve_img_dir('17660ef415d37059'))
        # self.save_ex1(1,"/mnt/d/Datasets/GoogleLandmarkRecognition2021/landmark-recognition-1perclass")

    def load_csv(self):
        print("Reading from train.csv")
        csv_path = os.path.join(self.data_path,"train.csv")
        df = pd.read_csv(csv_path)
        return df

    def create_dict(self,df):
        grouped = df.set_index("landmark_id").groupby(level=0)
        return dict(grouped.apply(lambda x: x.to_dict(orient='records')))

    def create_split(self):
        
        # Create directory if does not exist
        if not(os.path.exists(self.out_path)):
            os.makedirs(self.out_path)

        # Save 1 image from every class
        save_indices = self.classes.keys() #self.num_classes)]

        print(self.classes.keys())

        if(self.mp):
            print("Creating split with {} workers and saving to \"{}\"".format(num_workers,output_path))

            with multiprocessing.Pool(processes=int(self.num_workers)) as pool:
                data = pool.map(self.save_ex,save_indices)

                self.save_csv(data)
        else:
            print("Creating split with 1 worker and saving to \"{}\"".format(output_path))

            data = []
            for i in save_indices:
                data = self.save_ex(i)

            self.save_csv(data)

    def save_ex(self,lm_index:int,img_index:int=0):

        # Get the image id of the first landmark with landmark_id = id
        lm_id = self.get_lm_from_id(lm_index=lm_index,im_index=img_index)

        # Resolve image path from image id
        img_path = self.resolve_img_path(lm_id)
        img_dir = self.resolve_img_dir(lm_id)

        full_dir = os.path.join(self.out_path,img_dir)

        # Get image from full dataset and save image to new location
        if not(os.path.exists(full_dir)):
            os.makedirs(full_dir)

        shutil.copy(os.path.join(self.data_path,"train",img_path), full_dir)

        # Add image id to list of image id's in new dataset
        return ({lm_id: lm_index})

    def save_csv(self,data_dict):

        # Save csv location
        csv_path = os.path.join(self.out_path,"train.csv")

        header = ["id","landmark_id"]

        with open(csv_path,"w+") as f:
            writer = csv.writer
            writer.writerow(header)

            # Loop through examples in data_dict and save dict to csv
            for ex in data_dict:
                print(ex)
                for k, v in ex.items():
                    writer.writerow([k,v])

        
    def get_lm_from_id(self,lm_index:int,im_index:int=0):
        # Gets a image index (im_index) from a given landmark index (lm_index)
        print(lm_index,im_index)
        if(im_index>len(self.classes[lm_index])):
            raise ValueError("im_index exceeds # of images with this landmark id!")
        return self.classes[lm_index][im_index]["id"]

    def resolve_img_path(self,im_id:str):
        # image abcdef.jpg is placed in a/b/c/abcdef.jpg
        return os.path.join(im_id[0],im_id[1],im_id[2],str(im_id)+".jpg")

    def resolve_img_dir(self,im_id:str):
        # image abcdef.jpg is placed in a/b/c/abcdef.jpg
        return os.path.join(im_id[0],im_id[1],im_id[2])

if(__name__=="__main__"):
    # Load env variables from .env file
    cfg = dotenv_values(".env")
    
    # Set relevant variables
    dataset_path = cfg["DATASET_PATH"]
    num_workers = cfg["NUM_WORKERS"]

    output_path = cfg["OUTPUT_PATH"]

    # Create data_splitting object with dataset at dataset_path and num_workers
    d = Data_Splitting(dataset_path,output_path,num_workers)

    # Split data to a smaller dataset saved at output_path and with N samples
    d.create_split()
    



    