# Native Library Import
import json
import os
import shutil
import sys
import traceback
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
# Pip or Installed Library Import
from colorama import Fore, Style

# Custom file Import
from autoai_process import Config
from autoai_process.auto_ai_download import csv_to_json_convert, download_files
from autoai_process.builtin_func import auto_ai_connect
from autoai_process.gcp_train_utils import data_preprocessing
from training import test_model, train_model

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

class training_function():
    def __init__(self, model_details, retrain):
        self.model_details = model_details
        self.retrain = retrain

        # Paths can be updated, added, or removed based on developer requirements #

        # This is where training data would be stored
        self.dataset_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], "train data")
        # This is where the weights of the trained model would be stored
        self.models_path = os.path.join(
            Config.MODELS_PATH, self.model_details['_id'])
        # Path to csv file which contains info about the training data
        self.train_csv_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], self.model_details['resourcesFileName'])

        # This is where testing data would be stored
        self.test_dataset_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}", "test data")
        # Path to csv file which contains info about the testing data
        self.test_csv_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}",
            self.model_details['defaultDataSetCollectionResourcesFileName'])

    def train(self):
        print("============== model_details ==============")
        print(self.model_details)

        Config.MODEL_STATUS = 'Downloading Data'

        # Downloading the data from the csv file for training
        download_files(self.train_csv_path, self.dataset_path)
        download_files(self.test_csv_path, self.test_dataset_path)

        # If retrain == True than set the weight path and freeze some layers
        if self.retrain:
            retrain_weight = os.path.join(
                Config.MODELS_PATH, self.model_details['_id'], "checkpoint", 'modelDir')
        else:
            retrain_weight = None

        Config.MODEL_STATUS = "Training"

        print('Training started')

        # The call to start the training of the model should be made here #
        train_model.main(dataset_path=self.dataset_path,
                         models_path=self.models_path,
                         model_details=self.model_details)

        print('Training Completed')

        files_to_send = {
            'parentfile': [],
            'modelfile': [],
            'analyticfile': [],
            'additionalfile': []
        }

        for file in os.listdir(self.models_path):
            files_to_send['modelfile'].append(os.path.join(self.models_path, file))

        files_to_send['parentfile'].append(os.path.join(self.models_path, "best_model.h5"))

        # =============== Starting the test =============== #

        print("Testing Started")
        Config.MODEL_STATUS = 'Testing'

        test_detail = dict()
        try:
            print(self.model_details)
            print(self.model_details['defaultDataSetCollectionId'])
        except:
            self.model_details['defaultDataSetCollectionId'] = f"{random.randint(1111,10000000)}"

        if 'defaultDataSetCollectionResourcesFileGCStoragePath' in self.model_details.keys():
            test_detail['defaultDataSetCollectionId'] = self.model_details['defaultDataSetCollectionId']
            test_detail['defaultDataSetCollectionFileName'] = os.path.basename(
                self.model_details['defaultDataSetCollectionResourcesFileGCStoragePath']
            )
            try:
                # This is where the predictions of the model will be stored
                # output_file = os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", "results.csv")
                output_file = os.path.join(Config.DATA_PATH,
                                           self.model_details['_id'], f"test_{self.model_details['_id']}.csv")
                print(Fore.RED + "From training/train_func.py Line 139")
                print(Fore.RED + "The main() function present in the file training/test_model.py has to be updated based on developer requirements")

                # The call to start the testing of the model should be made here
                # The best performing model will be self.models_path/best_model.h5
                acc, confusion_matrix = test_model.main(test_csv_path=self.test_csv_path,
                                                          dataset_path=self.test_dataset_path,
                                                          models_path=os.path.join(self.models_path, "best_model.h5"),
                                                          output_file=output_file,
                                                          model_details=self.model_details)

                test_detail['accuracy'] = acc['Total']

                # Adding the test CSV
                files_to_send['analyticfile'].append(output_file)

                # Adding the test JSON
                with open(os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", f"test_{self.model_details['_id']}.json"), "w") as outfile:
                    json.dump(acc, outfile)

                files_to_send['analyticfile'].append(os.path.join(Config.DATA_PATH,
                                                                  f"test_{self.model_details['_id']}",
                                                                  f"test_{self.model_details['_id']}.json"
                                                                  )
                                                     )

                # Adding the confusion JSON
                with open(os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", f"confusion_{self.model_details['_id']}.json"), "w") as outfile:
                    json.dump(confusion_matrix, outfile)

                files_to_send['analyticfile'].append(os.path.join(Config.DATA_PATH,
                                                                  f"test_{self.model_details['_id']}",
                                                                  f"confusion_{self.model_details['_id']}.json"
                                                                  )
                                                     )


            except Exception:
                print(Fore.RED + "Test Failed")
                traceback.print_exc(file=sys.stdout)
                print(Style.RESET_ALL)

            # =============== Testing Completed =============== #

        print("Testing Completed")
        Config.MODEL_STATUS = 'Uploading Files'

        # Sending trained data to AutoAI
        auto_ai_connect.autoai_upload_files(
            url=Config.AUTOAI_URL_SEND_FILES,
            files=files_to_send,
            isDefault=True,
            id=self.model_details['_id'],
            test_detail=test_detail
        )

        Config.MODEL_STATUS = 'Training Completed'
        Config.POD_STATUS = "Available"

        print(Fore.GREEN + "Training Completed")
        print(Style.RESET_ALL)



