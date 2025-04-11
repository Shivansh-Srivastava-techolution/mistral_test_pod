import cv2
import numpy as np
# from fuzzywuzzy import process
import re
import io
import os
import json
from google.cloud import vision
import time
import json
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='vision_key.json'



class MatchingClass():
    def match_fuzzy_words(self, word, word_list, threshold=75):
        """
        Matches a word to similar fuzzy words in a given word list.

        Args:
            word (str): The word to match.
            word_list (list): A list of words to match against.
            threshold (int): The minimum similarity score required for a match.

        Returns:
            list: A list of tuples containing the matched word and its similarity score.
        """
        word_list_filtered = [word for word in word_list if len(word)>3]
        matches = process.extract(word, word_list_filtered, limit=None)
        fuzzy_matches = [(match[0], match[1]) for match in matches if match[1] >= threshold]
        print("All fuzzy matches above 75 for ",word,": ",fuzzy_matches)
        return fuzzy_matches if fuzzy_matches else None

    def apply_matching(self, text_list, task):
            search_keys_expiry_date = {
                'merit': expiry_date_database['merit'],
                'guidezilla': expiry_date_database['guidezilla'],
                'pilot': expiry_date_database['pilot'],
                'baylis': expiry_date_database['baylis'],
                'bardex': expiry_date_database['bardex']
            }
            if task=='expiry_date':
                fuzzy_matches_list = []
                total_highest_confidence = 0
                expiry_date = None
                index = -1
                confidence_of_match = 0
                for name, search_key_list in search_keys_expiry_date.items():
                    for search_key in search_key_list:
                        fuzzy_matches = self.match_fuzzy_words(search_key, text_list)
                        if fuzzy_matches:
                            current_total_confidence = sum(match[1] for match in fuzzy_matches)
                            # if any of match[1] for match in fuzzy_matches has value > 95 then lot_number = search_key and no need
                            # to loop further
                            if any(match[1] > 90 for match in fuzzy_matches):
                                expiry_date = search_key 
                                index = text_list.index(fuzzy_matches[0][0])
                                confidence_of_match = fuzzy_matches[0][1]
                                break     

                            if current_total_confidence > total_highest_confidence:
                                total_highest_confidence = current_total_confidence
                                expiry_date = search_key
                                index = text_list.index(fuzzy_matches[0][0])
                                confidence_of_match = fuzzy_matches[0][1]
                            fuzzy_matches_list.append(fuzzy_matches)
                return fuzzy_matches_list, expiry_date, index, confidence_of_match


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


def get_expiry_date_using_google_ocr(image_path):
    # image_path = os.path.join(dir_path, fn)
    st = time.time()
    op = cloud_vision_inference(image_path)
    time_taken = time.time() - st
    # res['image_path'] = op
    print("Total inference time for google ocr: ", time_taken)
    fn = os.path.basename(image_path)
    # res = []
    # res.append({fn:op})
    # print("OP: ",op)
    # print("-"*50)

    bboxes = []
    texts = []
    confidences = []

    for item in op:
        bbox = item['bbox']['vertices']
        bbox = [(bbox[0]['x'], bbox[0]['y']), 
                (bbox[1]['x'], bbox[1]['y']), 
                (bbox[2]['x'], bbox[2]['y']), 
                (bbox[3]['x'], bbox[3]['y'])]
        texts.append(item['text'])
        confidences.append(item['confidence'])
        bboxes.append(bbox)

    # print(word_list)
    matching_object = MatchingClass()

    fuzzy_matches_list_expiry_date, expiry_date, index_expiry_date, confidence_of_match_expiry_date = matching_object.apply_matching(texts, 'expiry_date')
    print(f"expiry date from gocr algo {expiry_date}")
    print(f"fuzzy mapping list {fuzzy_matches_list_expiry_date}")
    
    return expiry_date

    


# if __name__ == '__main__':
    
    
#     image_path = "/home/jupyter/Ownens_n_minors/OCR/ocr_catheter/dev2/google_ocr_pipeline/debug_plane_other/0_5344877122-image.png"
#     expiry_date = get_expiry_date_using_google_ocr(image_path)
#     print("Expiry Date: ", expiry_date)
    
    '''
    
    res = {}
    dir_path = "/home/jupyter/Ownens_n_minors/OCR/ocr_catheter/dev2/google_ocr_pipeline/merit_images/merit/"
    for fn in os.listdir(dir_path):
        image_path = os.path.join(dir_path, fn)
        st = time.time()
        op = cloud_vision_inference(image_path)
        time_taken = time.time() - st
        res['image_path'] = op
        print("Total inference time for google ocr: ", time_taken)
        fn = os.path.basename(image_path)
        # res = []
        # res.append({fn:op})
        # print("OP: ",op)
        print("-"*50)

        bboxes = []
        texts = []
        confidences = []

        for item in op:
            bbox = item['bbox']['vertices']
            bbox = [(bbox[0]['x'], bbox[0]['y']), 
                    (bbox[1]['x'], bbox[1]['y']), 
                    (bbox[2]['x'], bbox[2]['y']), 
                    (bbox[3]['x'], bbox[3]['y'])]
            texts.append(item['text'])
            confidences.append(item['confidence'])
            bboxes.append(bbox)


        word_list = []
        bbox_list = []
        conf_list = []
        for i, (text, bbox, conf) in enumerate(zip(texts, bboxes, confidences)):
            # print(i, text, round(conf*100))
            word_list.append(text)
            bbox_list.append(bbox)
            conf_list.append(conf)

        # print(word_list)
        matching_object = MatchingClass()

        fuzzy_matches_list_expiry_date, expiry_date, index_expiry_date, confidence_of_match_expiry_date = matching_object.apply_matching(word_list, 'expiry_date')


        json_object = {
            "filename": fn,
            "expiry_date_details": {
                "expiry_date": expiry_date,
                "bbox": bboxes[index_expiry_date] if index_expiry_date!=-1 else [],
                "confidence": (confidences[index_expiry_date]*100+confidence_of_match_expiry_date)/2
            },
            "time_taken": time_taken,
        }
        # No need as we are using ref number as well for class
        # if lot_number in lot_number_database['merit']:
        #     json_object['lot_number_details']['class'] = 'merit'
        # elif lot_number in lot_number_database['guidezilla']:
        #     json_object['lot_number_details']['class'] = 'guidezilla'
        # elif lot_number in lot_number_database['pilot']:
        #     json_object['lot_number_details']['class'] = 'pilot'
        # elif lot_number in lot_number_database['baylis']:
        #     json_object['lot_number_details']['class'] = 'baylis'
        # elif lot_number in lot_number_database['bardex']:
        #     json_object['lot_number_details']['class'] = 'bardex'
        # else:
        #     json_object['lot_number_details']['class'] = 'error'


        if expiry_date in expiry_date_database['merit']:
            json_object['expiry_date_details']['class'] = 'merit'
        elif expiry_date in expiry_date_database['guidezilla']:
            json_object['expiry_date_details']['class'] = 'guidezilla'
        elif expiry_date in expiry_date_database['pilot']:
            json_object['expiry_date_details']['class'] = 'pilot'
        elif expiry_date in expiry_date_database['baylis']:
            json_object['expiry_date_details']['class'] = 'baylis'
        elif expiry_date in expiry_date_database['bardex']:
            json_object['expiry_date_details']['class'] = 'bardex'
        else:
            json_object['expiry_date_details']['class'] = 'error'

        # print json_object with proper indent
        for key, value in json_object.items():
            print(f"{key}: {value}")
        print("-"*50)
        
        img = cv2.imread(image_path)
        cv2.putText(img, expiry_date, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.rectangle(img, bbox[0], bbox[2], (0, 0, 255), 3)
        
        cv2.imwrite("/home/jupyter/Ownens_n_minors/OCR/ocr_catheter/dev2/google_ocr_pipeline/merit_images/output/output_"+fn, img)
        
        
    with open('data.json', 'w') as fp:
        json.dump(res, fp)
        

    #     dir_path = "/home/jupyter/Ownens_n_minors/OCR/ocr_catheter/dev2/google_ocr_pipeline/debug_google_ocr"
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    #     cv2.imwrite("debug_google_ocr/"+os.path.basename(image_path), image)

    #     with open('debug_google_ocr/output_google_ocr_' + '.json', 'w') as f:
    #         if f.tell() != 0:  # Check if file is not empty, add comma for multiple JSON objects
    #             f.write(',\n')
    #         f.write(json.dumps(json_object, indent=4))


    #     ocr_task_id_rlef = "65c88062ddbdf68319a72fc6"
    #     send_to_rlef.send_image_to_rlef(request_id, "backlog", json.dumps(json_object), ocr_task_id_rlef, "google_ocr_1",
    #                                     "google_ocr_" + request_id,
    #                                     100, "predicted", "image",
    #                                     "debug_google_ocr/"+os.path.basename(image_path), "image/png")
    '''
