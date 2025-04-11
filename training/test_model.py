import os
import pandas as pd
# from keras.preprocessing import image
# from keras.models import load_model
# from difflib import SequenceMatcher
from PIL import Image
import torch
import glob, json
# from ultralytics import YOLO
import glob
from tqdm import tqdm
import numpy as np
import cv2, traceback
import cv2, time, os, glob, requests, datetime
import matplotlib.pyplot as plt
import time
from paddleocr import PaddleOCR
from numpy import array
from numpy import count_nonzero
from google.cloud import vision
import io
from torchmetrics.text import CharErrorRate
# from gocr_inference import cloud_vision_inference


def find_character_error_rate(predicted_text,actual_text):
    
    
    cer = CharErrorRate()
    u = cer(predicted_text, actual_text)

    return float(u)

# model = YOLO('/home/jupyter/background_rem/segmentation_training_pod/overlap_Removal5.pt')







#paddleocr class
class OCR():
    def __init__(self) -> None:
        self.model = PaddleOCR(use_angle_cls=True, lang='en')
    def ocr(self, img: str or np.ndarray) -> list:
        return self.model.ocr(img, cls = True)[0]



obj_ocr = OCR()


#googleocr class 
def cloud_vision_inference(path):
    """
    Performs cloud vision inference on the given image file.    

    Args:
        path (str): The path to the image file.

    Returns:
        list: A list of dictionaries containing the extracted text, bounding box vertices, and confidence level for each word in the image.
    """
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    data = response.full_text_annotation

    result_list = []

    for page in data.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                bbox = paragraph.bounding_box
                text = ''
                confidence = 1

                for word in paragraph.words:
                    for symbol in word.symbols:
                        text += symbol.text
                        if symbol.confidence < confidence:
                            confidence = symbol.confidence
                    text += ' '

                result_list.append({
                    "text": text,
                    "bbox": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in bbox.vertices
                        ]
                    },
                    "confidence": confidence
                })

    return result_list















def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def rotate_image(input_path, output_path, angle):
    # Open the image file
    image = Image.open(input_path)

    # Rotate the image
    rotated_image = image.rotate(angle)

    # Save the rotated image
    rotated_image.save(output_path)
    
def ocr_inference(image_path, gt_labels, countd):
    result = obj_ocr.ocr(image_path)
    records = []
    # categories = ['ltlatmid',
    #              'rtapex',
    #              'rtlatbase',
    #              'ltmid',
    #              'rtlatmid',
    #              'ltlatapex',
    #              'rtbase',
    #              'ltbase',
    #              'ltlatbase',
    #              'rtlatapex',
    #              'rtmid',
    #              'ltapex']
    raw_texts = []
    print("Actual Result")
    print(result)
    # if result==None:
        
        # result_col = []
        # status_col = result
        # for angle in [i for i  in range(360)][::15]:
        #     print("Calculating at angle : ", angle)
        #     output_path = "out.png"
        #     rotate_image(image_path, output_path, angle)
        #     result = obj_ocr.ocr(image_path)
        #     if result!=None:
        #         result_col.append(result)
        #         status_col = result
        #     print(result)
        # # if status_col==None:
        # #     return [], gt_labels, []
        # if result!=None:
        #     result = result_col[[np.mean([x[1][1] for x in resd]) for resd in result_col].index(max([np.mean([x[1][1] for x in resd]) for resd in result_col]))]
        #     print(result_col)
        #     print(resultd)
        #     print(categories)
        # import sys
        # sys.exit()
    if result==None:
        return [], [], []
    
        # result = []
    # print("="*30)
    # print("Frequency : ", countd)
    # print("="*30)
    # for resd, gt in zip(result, gt_labels):
    score = sum([x[1][1] for x in result])/len(result)
    score = round((score * 100),2) 
    label = ""
    for z in result:
        label += z[1][0]
    raw_text = " ".join(label.split(" "))
    print("="*20)
    print("Predicted Data : ", result)
    print("Raw Data : ", raw_text)
    raw_texts.append(raw_text)
    max = -1
    targ = ""
    # for ctg in categories:
    #     score = similar(raw_text, ctg)
    #     if score>max:
    #         targ = ctg
    #         max = score
    #if targ==gt:
    
    cer_score = find_character_error_rate(raw_text,gt_labels)
    
    # print(f"================CER : {cer_score}======================")
    
    cer_score = 100 - round(cer_score,2)
    # print(f"================CER : {cer_score}======================")
    
    final_score = (score + cer_score) / 2
    
    final_score = round(final_score,2)
    
    print("score after average: ",final_score)
    
    labeld = raw_text
    
    
    print("Caculated Data : ", targ)
    print("="*20)
    #else:
    #    label = raw_text
    # print(image_path)
    # print(resd)
    # minx = np.min([int(x[0]) for x in resd[0]])
    # miny = np.min([int(x[1]) for x in resd[0]])
    # maxx = np.max([int(x[0]) for x in resd[0]])
    # maxy = np.max([int(x[1]) for x in resd[0]])
    # coords = [minx, miny, maxx, maxy]
    coords = [f[0] for f in result]
    records.append([coords, labeld, score])
    
     
    return records, raw_texts, final_score


def gocr_inference(image_path, gt_labels, countd):
    result = cloud_vision_inference(image_path)
    records = []
    
    raw_texts = []
    print("Actual Result")
    print(result)

    if result==None:
        return [], [], []
    
    # result_list.append({
    #                 "text": text,
    #                 "bbox": {
    #                     "vertices": [
    #                         {"x": vertex.x, "y": vertex.y}
    #                         for vertex in bbox.vertices
    #                     ]
    #                 },
    #                 "confidence": confidence
    #             })

    score = [det["confidence"] for det in result]
    score = sum(score)/len(score)
    score = round((score * 100),2) 
    # label = ""
    # for det in result:
    #     label += det["text"]
    label = result[0]["text"]
    if label.find("-") >= 0:
        label = "".join(label.split(" "))
    label = label.strip()
    raw_text = label
    print("="*20)
    print("Predicted Data : ", result)
    print("Raw Data : ", raw_text)
    raw_texts.append(raw_text)
    max = -1
    targ = ""
    
    
    cer_score = find_character_error_rate(raw_text,gt_labels)
    
    # print(f"================CER : {cer_score}======================")
    
    cer_score = 100 - round(cer_score,2)
    # print(f"================CER : {cer_score}======================")
    
    final_score = (score + cer_score) / 2
    
    final_score = round(final_score,2)
    
    print("score after average: ",final_score)
    
    labeld = raw_text
    
    
    print("Caculated Data : ", targ)
    print("="*20)
    #else:
    #    label = raw_text
    # print(image_path)
    # print(resd)
    # minx = np.min([int(x[0]) for x in resd[0]])
    # miny = np.min([int(x[1]) for x in resd[0]])
    # maxx = np.max([int(x[0]) for x in resd[0]])
    # maxy = np.max([int(x[1]) for x in resd[0]])
    # coords = [minx, miny, maxx, maxy]
    coords = [f["bbox"]["vertices"] for f in result]
    records.append([coords, labeld, score])
    
     
    return records, raw_texts, final_score








def main(test_csv_path, dataset_path, models_path, output_file, model_details):
    """
    dataset_path: the path where testing data is stored
    models_path: the path where the weights of the trained model is stored
    output_file: the path where the predictions of the model will be stored

    The main objective of this function is to test the model using the weights present at the path "models_path" and on the data present in the "dataset_path".
    This function should store it's prediction into the file "output_file", in a csv format. You can look at example_results.csv to gain more clarity about it.
    This function should return a dictionary containing overall accuracy and also accuracy for specific labels. You can look at example_acc_dict.json to gain more clarity about it.

    """
#     actuals, names = [], []
#     for file in glob.glob(f"{test_folder}/*/*.*p*g"):
#         actuals.append(file.split('/')[-2])
#         names.append(file.split('/')[-1].split('.')[0])
#     predicted, final_scores = [], []
#     for filex in glob.glob(f"{test_folder}/*/*.*p*g"):
#         _id = datetime.datetime.now().timestamp()
#         img = cv2.imread(filex)
#         framex = img[:, :int(img.shape[1]/2)].copy()
#         result1 = model(framex)
#         targ1 = result1[0].masks.data[0].cpu().numpy()*255
#         for mask in result1[0].masks.data:
#             targ1+=mask.cpu().numpy()*255
#         cv2.imwrite("resd.png", targ1)
#         targ1 = cv2.imread("resd.png")
#         targ1 = cv2.cvtColor(targ1, cv2.COLOR_BGR2GRAY)
#         classes = sorted([x.split("/")[-1] for x in glob.glob(f"{test_folder}/*")])
#         framey = img[:, -int(img.shape[1]/2):].copy()
#         result2 = model(framey)
#         targ2 = result2[0].masks.data[0].cpu().numpy()*255
#         for mask in result2[0].masks.data:
#             targ2+=mask.cpu().numpy()*255
#         cv2.imwrite("resd.png", targ2)
#         targ2 = cv2.imread("resd.png")
#         targ2 = cv2.cvtColor(targ2, cv2.COLOR_BGR2GRAY)
#         resd = []
#         for z in (targ1-targ2):
#             for k in z:
#                 if k>40:
#                     resd.append(k)
#         target = (targ1-targ2)
#         A = target.copy()
#         A = np.nan_to_num(A,0)
#         sparsity = 1.0 - ( count_nonzero(A) / float(A.size) )
#         # print(sparsity)
#         # print(len(resd))
#         # print(abs(len(result1[0].masks.data)- len(result2[0].masks.data)))
#         # if len(resd)>400:
#         score = sparsity
#         csv = f"""Pixel Intensity - {len(resd)}<br>
#                   Mask_Count - {abs(len(result1[0].masks.data)- len(result2[0].masks.data))}<br>
#                   Image File - {filex.split("/")[-1]}<br>
#                """
#         # combined = np.concatenate((framex, framey), axis = 1)
#         # cv2.imwrite(f"{_id}_raw_image.png", combined)
#         if len(resd)>400 and sparsity<0.98 and (abs(len(result1[0].masks.data)- len(result2[0].masks.data)))>0:
#             # file_path = "combined.png"
#             label = "SKU_Changed"
#             score = sparsity
#             final_scores.append([(1-score), score])
#             # break
#         else:
#             # file_path = "combined.png"
#             label = "No_Change"
#             score = sparsity
#             final_scores.append([score, (1-score)])

#         predicted.append(label)

    #============
    # model = load_model(models_path)

    df = pd.read_csv(test_csv_path)
    resource_ids = df['_id']
    gcs_filenames = df['GCStorage_file_path']
    actual_labels = df['label']
    # target_labels = ['ltlatmid',
    #                  'rtapex',
    #                  'rtlatbase',
    #                  'ltmid',
    #                  'rtlatmid',
    #                  'ltlatapex',
    #                  'rtbase',
    #                  'ltbase',
    #                  'ltlatbase',
    #                  'rtlatapex',
    #                  'rtmid',
    #                  'ltapex',
    #                  'unknown']
    
    
    columns = ['resource_id', 'trim_id', 'filename', 'label', 'parentLabel', 'predicted', 'confidence_score']+['startTime', 'endTime']
    output_df = pd.DataFrame(
        columns=columns)

    # total_Items_Present = 0
    # total_No_Items = 0
    # correct_Items_Present = 0
    # correct_No_Items = 0

    # confusion_matrix = {'Items_Present': {'Items_Present': 0, 'No_Items': 0}, 'No_Items': {'Items_Present': 0, 'No_Items': 0}}
    confusion_matrix = {}
    for label in actual_labels:
        confusion_matrix[label] = {}
        for labeld in actual_labels:
            confusion_matrix[label][labeld] = 0
    
    # for actual_label in actual_labels:
    #     confusion_matrix[actual_label] = {}
    #     for actual_labeld in actual_labels:
    #         confusion_matrix[actual_label][actual_labeld] = 0

    records_inf = {}
    count = 0
    # records_inf = []
    for resource_id, gcs_filename, actual_label in zip(resource_ids, gcs_filenames, actual_labels):
        count+=1
        # Load an image file to test, resizing it to 150x150 pixels (as required by this model)

        # If the hyperparameter is coming during training

        image_path = os.path.join(dataset_path, actual_label, os.path.basename(gcs_filename))
#         img = cv2.imread(image_path)
#         framex = img[:, :int(img.shape[1]/2)].copy()
#         result1 = model(framex)
#         targ1 = result1[0].masks.data[0].cpu().numpy()*255
#         for mask in result1[0].masks.data:
#             targ1+=mask.cpu().numpy()*255
#         cv2.imwrite("resd.png", targ1)
#         targ1 = cv2.imread("resd.png")
#         targ1 = cv2.cvtColor(targ1, cv2.COLOR_BGR2GRAY)
#         # classes = sorted([x.split("/")[-1] for x in glob.glob(f"{test_folder}/*")])
#         framey = img[:, -int(img.shape[1]/2):].copy()
#         result2 = model(framey)
#         targ2 = result2[0].masks.data[0].cpu().numpy()*255
#         for mask in result2[0].masks.data:
#             targ2+=mask.cpu().numpy()*255
#         cv2.imwrite("resd.png", targ2)
#         targ2 = cv2.imread("resd.png")
#         targ2 = cv2.cvtColor(targ2, cv2.COLOR_BGR2GRAY)
#         resd = []
#         for z in (targ1-targ2):
#             for k in z:
#                 if k>40:
#                     resd.append(k)
#         target = (targ1-targ2)
#         A = target.copy()
#         A = np.nan_to_num(A,0)
#         sparsity = 1.0 - ( count_nonzero(A) / float(A.size) )
#         # print(sparsity)
#         # print(len(resd))
#         # print(abs(len(result1[0].masks.data)- len(result2[0].masks.data)))
#         # if len(resd)>400:
#         # score = sparsity
#         # csv = f"""Pixel Intensity - {len(resd)}<br>
#         #           Mask_Count - {abs(len(result1[0].masks.data)- len(result2[0].masks.data))}<br>
#         #           Image File - {filex.split("/")[-1]}<br>
#         #        """
#         # combined = np.concatenate((framex, framey), axis = 1)
#         # cv2.imwrite(f"{_id}_raw_image.png", combined)
#         if len(resd)>400 and sparsity<0.98 and (abs(len(result1[0].masks.data)- len(result2[0].masks.data)))>0:
#             # file_path = "combined.png"
#             label = "SKU_Changed"
#             score = sparsity
#             # final_scores.append([(1-score), score])
#             # break
#         else:
#             # file_path = "combined.png"
#             label = "No_Change"
#             score = sparsity
#             # final_scores.append([score, (1-score)])

#         # predicted.append(label)


        # For binary classification, the prediction will be a float between 0-1. You can convert this to a label
        # predicted_label = 'dogs' if prediction > 0.5 else 'cats'
        # result = model(image_path)
        # if len(result[0].boxes.conf.cpu().numpy())>0:
        #     label = "Items_Present"
        #     score = np.mean(result[0].boxes.conf.cpu().numpy())
        # else:
        #     label = "No_Items"
        #     score = 1
        records, raw_texts, scores = gocr_inference(image_path, [actual_label], count)
        
        
        print("="*4, records, "="*4)
        print(scores)
        print(raw_texts)
        if len(records)!=0:
            # print(image_path)
            # print(scores)
            # print(scores.index(max(scores)), len(records))
            # print(records[scores.index(max(scores))])
            # label = records[scores.index(max(scores))][1]
            label = raw_texts[0]
            score = scores
            prediction = label
            predicted_label = label
        else:
            label = 'unknown'
            score = 0
            prediction = label
            predicted_label = label
        # Add the data to the output dataframe
        try:
            meta_data = json.dumps({"raw_text":raw_texts[0]})
        except:
            meta_data = json.dumps({"raw_text":''})
            
        append_df = {'resource_id': resource_id,
                                      'trim_id': '',
                                      'filename': os.path.basename(gcs_filename),
                                      'label': actual_label,
                                      'parentLabel': actual_label,
                                      'predicted': label,
                                      'confidence_score': round(score, 2),
                                      
                                      # f'label_{actual_label}': round(score, 2) if actual_label == 'No_Items' else 0,
                                      'startTime': 0.0,
                                      'endTime': 1.0, 
                                      'metadata':meta_data,
                                      f'label_{actual_label}': 100 if actual_label == label else 0
                                      }
        
        print("Final Label : ", label)
        print("Final Score : ", score)
        # for z in target_labels:
        #     append_df[f'label_{z}'] = round(score*100, 2) if z==label else 0
        records_inf[actual_label] = label
        output_df = output_df._append(append_df, ignore_index=True)
        # if count==10:
        #     break
        # Update the accuracy
        # if actual_label == 'Items_Present':
        #     total_Items_Present += 1
        #     if predicted_label == 'Items_Present':
        #         correct_Items_Present += 1
        # elif actual_label == 'No_Items':
        #     total_No_Items += 1
        #     if predicted_label == 'No_Items':
        #         correct_No_Items += 1

        # Updating the confusion matrix
        print("="*30)
        print(actual_label)
        print(predicted_label)
        try:
        # if predicted_label == actual_label:
            confusion_matrix[actual_label][predicted_label] = confusion_matrix[actual_label][predicted_label] + 1
            # print("updated value: ",confusion_matrix[actual_label][predicted_label])
            print("****************** 1st case in confusion matrix ***************************")
            
#         elif confusion_matrix[actual_label][predicted_label] == 1:
#             confusion_matrix[actual_label][predicted_label] += 1
#             print("****************** 3rd case in confusion matrix ***************************")
            
#         else:
#             confusion_matrix[actual_label][predicted_label] = 1
#             print("****************** 2nd case in confusion matrix ***************************")
        except:
            print("it came here")
            # print("****************** 2nd case in confusion matrix ***************************")
            # confusion_matrix[str(actual_label)][str(predicted_label)] = 1
            
    
    # Calculate the accuracy
    df_rec = pd.DataFrame(records_inf.items(), columns=['actual_label', 'predicted'])
    
    acc = {'Total':round((df_rec[df_rec['actual_label']==df_rec['predicted']].shape[0]/df_rec.shape[0]) * 100, 2)}

    
    name_counts = output_df['label'].value_counts().reset_index()
    name_counts.columns = ['label', 'Count']

    # Merge count with original DataFrame
    df1 = output_df.merge(name_counts, on='label')

    # Group by 'Name' and aggregate with sum
    grouped_df = df1.groupby('label').agg({'confidence_score': ['sum',"mean"]}).reset_index()
    grouped_df.columns = ['label', 'Total_confidence', "Average_confidence"]
    
    for index in range(len(grouped_df)):
        acc[grouped_df["label"][index]] = grouped_df["Average_confidence"][index]

    
    
    

    for col in output_df.columns.tolist():
        if col.find("label_") >= 0:
            print("it came to fill nul values")
            output_df[[col]] = output_df.loc[:,[col]].fillna(value=0)

    print("Output Path : ", output_file)
    
    output_df.to_csv(output_file)

    
    return acc, confusion_matrix


if "__main__" == __name__:
    test_csv_path = "/home/jupyter/background_rem/pod_updated/add_rem_sku/testing_pod/Keras-Applications-Pod/data/test_6575c131f5e454c3bb04ca22/defaultDataSetCollection_65753d843b58d84c89eaeded_resources.csv"
    dataset_path = "/home/jupyter/background_rem/pod_updated/add_rem_sku/testing_pod/Keras-Applications-Pod/data/test_6575c131f5e454c3bb04ca22/test data"
    models_path = "/home/anandhakrishnan/Projects/American-Ordinance/ao-fedramp/training-pods/Yolo-V58-Pod/runs/detect/train/weights/best.pt"
    output_file = "temp123.csv"
    statistics_file = "temp123.json"
    model_details = {}
    # main(test_csv_path, test_json_path, dataset_path, models_path, output_file, statistics_file, hyperparameters)
    main(test_csv_path, dataset_path, models_path, output_file, model_details)
