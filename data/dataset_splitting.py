from dataset_transforms import Dataset_Transforms

from PIL import Image

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
        self.imgs_per_class = self.get_class_lengths()

        # Enable multiprocessing
        self.mp = False

        # Enable preprocessing
        # Preprocessing includes resize and centercrop
        self.preprocess = True

        if(self.preprocess):
            self.dt = Dataset_Transforms()
            self.transforms = self.dt.get_data_transforms()


    def load_csv(self):
        print("Reading from train.csv")
        csv_path = os.path.join(self.data_path,"train.csv")
        df = pd.read_csv(csv_path)
        return df

    def create_dict(self,df):
        grouped = df.set_index("landmark_id").groupby(level=0)
        return dict(grouped.apply(lambda x: x.to_dict(orient='records')))

    def get_class_lengths(self):
        class_lengths = {}

        # Get the lengths (number of image samples) of all classes
        for k, v in self.classes.items():
            class_lengths[k] = len(v)

        return {k: v for k, v in sorted(class_lengths.items(),key=lambda item:item[1],reverse=True)}

    # Split dataset with 1 image from each class (~81.3k samples)
    def create_split_1pc(self):
        
        # Create directory if does not exist
        if not(os.path.exists(self.out_path)):
            os.makedirs(self.out_path)

        # Save 1 image from every class
        save_indices = self.classes.keys() #self.num_classes)]

        # print(self.classes.keys())

        if(self.mp):
            print("Creating split with {} workers and saving to \"{}\"".format(num_workers,output_path))

            with multiprocessing.Pool(processes=int(self.num_workers)) as pool:
                data = pool.map(self.save_ex,save_indices)

                self.save_csv(data,"train")
        else:
            print("Creating split with 1 worker and saving to \"{}\"".format(output_path))

            data = []
            for i in save_indices:
                data.append(self.save_ex(i))

            self.save_csv(data,"train")

    # Split dataset with 1000 classes, all images from class
    def create_split_1k(self):
        
        # Create directory if does not exist
        if not(os.path.exists(self.out_path)):
            os.makedirs(self.out_path)

        # print(self.classes)
        save_indices = list(self.imgs_per_class.keys())[:1000]

        if(self.mp):
            print("Creating split with {} workers and saving to \"{}\"".format(num_workers,output_path))

            with multiprocessing.Pool(processes=int(self.num_workers)) as pool:
                data = pool.map(self.save_ex_n,save_indices)

                data = [item for sublist in data for item in sublist]
                self.save_csv(data,"train")
        else:
            print("Creating split with 1 worker and saving to \"{}\"".format(output_path))

            data = []
            for i in save_indices:
                data.append(self.save_ex_n(i))
            data = [item for sublist in data for item in sublist]
            self.save_csv(data,"train")

    def create_split_1k_preprocess(self):
        
        if not(self.preprocess):
            print("Error: preprocess set to False")
            return

        # Create directory if does not exist
        if not(os.path.exists(self.out_path)):
            os.makedirs(self.out_path)

        # print(self.classes)
        save_indices = list(self.imgs_per_class.keys())[:1000]

        if(self.mp):
            print("Creating split with {} workers and saving to \"{}\"".format(num_workers,output_path))

            with multiprocessing.Pool(processes=int(self.num_workers)) as pool:
                data = pool.map(self.process_img_n,save_indices)

                data = [item for sublist in data for item in sublist]
                self.save_csv(data,"train")
        else:
            print("Creating split with 1 worker and saving to \"{}\"".format(output_path))

            data = []
            for i in save_indices:
                data.append(self.process_img_n(i))
            data = [item for sublist in data for item in sublist]
            self.save_csv(data,"train")

    def create_split_test(self):
            
        self.out_path = output_path+str("-test")
    
        # Create directory if does not exist
        if not(os.path.exists(self.out_path)):
            os.makedirs(self.out_path)

        # print(self.classes)
        test_classes = 2
        save_indices = list(self.imgs_per_class.keys())[:2]

        if(self.mp):
            print("Creating split with {} workers and saving to \"{}\"".format(num_workers,output_path))

            with multiprocessing.Pool(processes=int(self.num_workers)) as pool:
                data = pool.map(self.process_img_n,save_indices)

                data = [item for sublist in data for item in sublist]
                self.save_csv(data,"train")
        else:
            print("Creating split with 1 worker and saving to \"{}\"".format(output_path))

            data = []
            for i in save_indices:
                data.append(self.process_img_n(i))
            data = [item for sublist in data for item in sublist]
            self.save_csv(data,"train")

    def process_img(self,lm_index:int,img_index:int=0):
        # Get the image id of the ith landmark (where i=img_index) 
        # and landmark_id = lm_index
        lm_id = self.get_lm_from_id(lm_index=lm_index,im_index=img_index)

        # Resolve image path from the image id
        img_path = self.resolve_img_path(lm_id)
        full_path = os.path.join(self.data_path,"train",img_path)

        # Path to save image example to
        # We save each image in a folder with name landmark_id
        save_path = os.path.join(self.out_path,str(lm_index))

        # Get image from full dataset and save image to new location
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)

        img = Image.open(full_path)
        img = self.transforms(img)
        img.save(os.path.join(save_path,str(lm_id)+".jpg"))


        # Add image id to list of image id's in new dataset
        return ({lm_id: lm_index})

    def process_img_n(self,lm_index:int,num_max=5):

        print("Processing and saving samples for class "+str(lm_index))
        
        ex_list = []
        num_imgs = min(num_max,self.imgs_per_class[lm_index])

        # Loop through all examples for a given landmark id and save it
        for i in range(0,num_imgs):
            ex_list.append(self.process_img(lm_index=lm_index,img_index=i))

        # Add list of  image id to list of image id's in new dataset
        return ex_list

    def save_ex(self,lm_index:int,img_index:int=0):

        # Get the image id of the ith landmark (where i=img_index) 
        # and landmark_id = lm_index
        lm_id = self.get_lm_from_id(lm_index=lm_index,im_index=img_index)

        # Resolve image path from the image id
        img_path = self.resolve_img_path(lm_id)

        # Path to save image example to
        # We save each image in a folder with name landmark_id
        save_path = os.path.join(self.out_path,str(lm_index))

        # Get image from full dataset and save image to new location
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)

        shutil.copy(os.path.join(self.data_path,"train",img_path), save_path)

        # Add image id to list of image id's in new dataset
        return ({lm_id: lm_index})

    def save_ex_n(self,lm_index:int,num_max=5):
        
        print("Saving samples for class "+str(lm_index))
        
        ex_list = []
        num_imgs = min(num_max,self.imgs_per_class[lm_index])

        # Loop through all examples for a given landmark id and save it
        for i in range(0,num_imgs):
            ex_list.append(self.save_ex(lm_index=lm_index,img_index=i))

        # Add list of  image id to list of image id's in new dataset
        return ex_list

    def save_ex_all(self,lm_index:int):

        ex_list = []

        # Loop through all examples for a given landmark id and save it
        for i in range(0,self.imgs_per_class[lm_index]):
            ex_list.append(self.save_ex(lm_index=lm_index,img_index=i))

        # Add list of  image id to list of image id's in new dataset
        return ex_list

    def save_csv(self,data_dict,file_name):

        # Save csv location
        csv_path = os.path.join(self.out_path,str(file_name)+".csv")

        header = ["id","landmark_id"]

        with open(csv_path,"w+") as f:
            w = csv.writer(f)
            w.writerow(header)

            # Loop through examples in data_dict and save dict to csv
            for ex in data_dict:
                # print(ex)
                for k, v in ex.items():
                    w.writerow([k,v])

        
    def get_lm_from_id(self,lm_index:int,im_index:int=0):
        # Gets a image index (im_index) from a given landmark index (lm_index)
        # print(lm_index,im_index)
        if(im_index>len(self.classes[lm_index])):
            raise ValueError("im_index exceeds # of images with this landmark id!")
        return self.classes[lm_index][im_index]["id"]

    def resolve_img_path(self,im_id:str):
        # image abcdef.jpg is placed in a/b/c/abcdef.jpg
        return os.path.join(im_id[0],im_id[1],im_id[2],str(im_id)+".jpg")

    def resolve_img_dir(self,im_id:str):
        # image abcdef.jpg is placed in a/b/c/abcdef.jpg
        return os.path.join(im_id[0],im_id[1],im_id[2])

def create_examples(path,save_path):
        """
        Appends additional training examples to existing dataset. Adds training example names to existing .csv file

        Parameters
        ----------
        path : str
            path to additional training examples
        """

        data = []

            # img = Image.open(full_path)
            # img = self.transforms(img)
            # img.save(os.path.join(save_path,str(filename)))

        print("Creating new data examples from folder "+str(path))
        print("Saving to "+save_path)

        for subdir, dirs, files in os.walk(path):
            for file in files:
                # print(subdir.split("/")[-1])
                # print(file.replace(".jpg",""))
                lm_id = subdir.split("/")[-1]
                lm_index = file.replace(".jpg","")

                full_path = os.path.join(subdir, file)
                tmp_spath = os.path.join(save_path, lm_id)

                if not(os.path.exists(tmp_spath)):
                    os.makedirs(tmp_spath)

                img = Image.open(full_path)
                dt = Dataset_Transforms()
                transforms = dt.get_data_transforms()
                img = transforms(img)
                img.save(os.path.join(save_path,lm_id,str(lm_index)+".jpg"))
                # print(os.path.join(subdir, file))
                data.append([lm_id, lm_index])

        with open(os.path.join(save_path,'train.csv'),'a') as file:
            writer = csv.writer(file)
            writer.writerows(data)

if(__name__=="__main__"):
    
    # Set relevant variables
    dataset_path = "/mnt/d/Datasets/GoogleLandmarkRecognition2021/landmark-recognition-2021"
    num_workers = 16
    output_path = "/mnt/d/Datasets/GoogleLandmarkRecognition2021/landmark-recognition-1k-5-preprocessed"

    # Create data_splitting object with dataset at dataset_path and num_workers
    d = Data_Splitting(dataset_path,output_path,num_workers)

    # Split data to a smaller dataset saved at output_path and with N samples
    # d.create_split_1k()
    # d.save_csv([{123:"asd1qgasf"},{245:"agawfgaxscva"},{456:"asfhqowifh"}],"test")

    # d.create_split_1k_preprocess()
    new_examples_path = "/mnt/d/Datasets/APS360/Train Photos"
    save_path = "/mnt/d/Datasets/APS360/NewTrainExamples"
    create_examples(new_examples_path,save_path)



    