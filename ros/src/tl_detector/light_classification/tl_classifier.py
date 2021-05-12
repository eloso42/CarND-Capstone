from styx_msgs.msg import TrafficLight
from keras import models
import tensorflow as tf
import numpy as np
import cv2


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = models.load_model('model.h5')
        global graph
        graph = tf.get_default_graph()
        print("ok")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.resize(image, (400,300))
        image = np.array([image])
        with graph.as_default():
            pred = self.model.predict(image, batch_size=1)
        print (pred)
        return np.argmax(pred)
        #TODO implement light color prediction
        #return TrafficLight.UNKNOWN
