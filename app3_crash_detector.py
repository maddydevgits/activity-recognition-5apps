import streamlit as st
import cv2
from constants import *
import numpy as np

from itertools import chain 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from madblocks import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AccidentsClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = 'graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            # Works up to here.
            with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded}
            )
            """(scores, classes) = self.sess.run(
                [self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded}
            )"""
        return boxes, scores, classes, num


def load_image(img_file):
    img=Image.open(img_file)
    return img

def app():
    st.title('Vehicle Crash Detector')
    src_file=st.file_uploader('upload image',type=['png','jpg','jpeg'])
    if src_file is not None:
        st.image(load_image(src_file)) 
        
        with open('crash/input.jpg','wb') as f:
            f.write(src_file.getbuffer())

        FRAME_WINDOW = st.image([])
                  
        PATH_TO_LABELS = 'accidents.pbtxt'
        NUM_CLASSES = 1

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        img = plt.imread('crash/input.jpg')#[::-1,:,::-1]
        img.setflags(write=1)

        x = AccidentsClassifier()

        boxes, scores, classes, num = x.get_classification(img)
        vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
        plt.imsave('crash/output.jpg', img)
        if(scores[0][0]>0.6):
            send_email_with_attachment(username="saimadhav.alert@gmail.com",password="m@keskilled",receiver="parvathanenimadhu@gmail.com",subject="Vehicle Crash Alert",body="This alert is generated automatically",imagefile="crash/output.jpg")

        st.image(load_image('crash/output.jpg'))
        
     