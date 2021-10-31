from dotenv import dotenv_values


import multiprocessing
import numpy as np
import os


class Data_Splitting():

    def __init__(dataset_path:str, num_workers:int=8):
        print(dataset_path)
        print(num_workers)

    def split(output_path:str):
        print(output_path)



if(__name__=="__main__"):
    # Load env variables from .env file
    cfg = dotenv_values(".env")
    
    # Set relevant variables
    dataset_path = cfg["DATASET_PATH"]
    num_workers = cfg["NUM_WORKERS"]

    output_path = cfg["OUTPUT_PATH"]

    # Create data_splitting object with dataset at dataset_path and num_workers
    d = Data_Splitting(dataset_path,num_workers)

    # Split data to a smaller dataset saved at output_path and with N samples
    d.split(output_path)

    