import numpy as np
import datetime
import time

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3"'
                      ' subdirectory if required)')

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

# parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network '
#                                  'trained either in Caffe or TensorFlow frameworks.')
# parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
# parser.add_argument("--prototxt", default="Mdels_detection_caffe/MobileNetSSD_deploy.prototxt.txt",
#                     help='Path to text network file: MobileNetSSD_deploy.prototxt for Caffe model or '
#                     'ssd_mobilenet_v1_coco.pbtxt from opencv_extra for TensorFlow model')
# parser.add_argument("--weights", default="Mdels_detection_caffe/MobileNetSSD_deploy.caffemodel",
#                     help='Path to weights: MobileNetSSD_deploy.caffemodel for Caffe model or '
#                                           'frozen_inference_graph.pb from TensorFlow.')

# parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
# args = parser.parse_args()


net = cv.dnn.readNetFromCaffe('Waights/Caffe/MobileNetSSD_deploy.prototxt.txt',
                              'Waights/Caffe/MobileNetSSD_deploy.caffemodel')
swapRB = False
classNames = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


class VideoCamera(object):
    def __init__(self):
        # 0 - web camera of computer, 1 - connecting web camera USB
        # or set IP path for IP camera
        self.cap = cv.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        success, image = self.cap.read()
        blob = cv.dnn.blobFromImage(image, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        now = datetime.datetime.now().strftime('%Y-%d-%b %H:%M:%S')

        cols = image.shape[1]
        rows = image.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        image = image[y1:y2, x1:x2]

        cols = image.shape[1]
        rows = image.shape[0]

        # sum of detection of person
        sum_person = 0
        # confidence threshold to filter out weak detections
        thr = 0.5
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > thr:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 255, 0))

                # if class == 15 is class_id == person then count persons in the image
                if class_id == 15:
                    sum_person += 1

                # sum_person_now = 'Persons: ' + str(sum_person)
                # print('Persons: ', sum_person)

                font = cv.FONT_HERSHEY_SIMPLEX
                #  Write time on img
                cv.putText(image, now, (10, 460), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                # #  Write sum of persons on img
                # cv.putText(image, sum_person_now, (10, 35), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)

                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, font, 0.5, 2)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv.FILLED)
                    cv.putText(image, label, (xLeftBottom, yLeftBottom),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        #  Write sum of persons on img
        sum_person_now = 'Persons: ' + str(sum_person)
        cv.putText(image, sum_person_now, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()

    def gen(self):
        while True:
            frame = self.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    #  ----------------------------------------Test Down-----------------------------------------
    # def count(self):
    #     while True:
    #         people_count = self.get_frame.sum_person
    #         yield (people_count)
