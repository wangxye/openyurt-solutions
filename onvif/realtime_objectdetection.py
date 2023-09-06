import collections
import os
import sys
import time

import cv2
import numpy as np
from IPython import display
from openvino.runtime import Core
import threading
from videoplay import VideoPlayer

'''
person-vehicle-bike-detection-2000
'''
base_model_dir = "models"
# model name as named in Open Model Zoo
model_name = "person-vehicle-bike-detection-2000-FP32"
# model_name = "ssdlite_mobilenet_v2"
# output path for the conversion
converted_model_path = f"{base_model_dir}/{model_name}.xml"

'''
Load the Model
我们将模型下载下来并转换成IR模型后，加载模型
'''
# initialize inference engine
ie_core = Core()
# read the network and corresponding weights from file
model = ie_core.read_model(model=converted_model_path)
# compile the model for the CPU (you can choose manually CPU, GPU, MYRIAD etc.)
# or let the engine choose the best available device (AUTO)
compiled_model = ie_core.compile_model(model=model, device_name="CPU")
print("3 - Load model and compile model.")
# get input and output nodes
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print("- Input layer info: ", input_layer)
print("- Output layer info: ", output_layer)
# get input size
# height, width = list(input_layer.shape)[1:3]
height, width = list(input_layer.shape)[2:4]

'''
接下来，我们列出所有可用的类并为它们创建颜色。 然后，在后处理阶段，我们将归一化坐标为`[0, 1]`的框转换为像素坐标为`[0, image_size_in_px]`的框。
之后，我们使用非最大抑制来删除重叠框以及低于阈值为0.5的框。最后，我们可以将剩下的绘制框和标签绘制在视频中。
'''

# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# classes = [
#     "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
#     "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
#     "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
#     "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
#     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
#     "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
#     "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
#     "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
#     "teddy bear", "hair drier", "toothbrush", "hair brush"
# ]

classes = [
  "vehicle","person","bike"
]

# colors for above classes (Rainbow Color Map)
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()
print("4 - 我们列出所有可用的类并为它们创建颜色。")

print("5 - 我们使用非最大抑制来删除重叠框以及低于阈值为0.5的框。最后，我们可以将剩下的绘制框和标签绘制在视频中。")
def process_results(frame, results, thresh=0.6):
    # size of the original frame
    h, w = frame.shape[:2]
    # results is a tensor [1, 1, 100, 7]
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # create a box with pixels coordinates from the box with normalized coordinates [0,1]
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    # apply non-maximum suppression to get rid of many overlapping entities
    # see https://paperswithcode.com/method/non-maximum-suppression
    # this algorithm returns indices of objects to keep
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
    )

    # if there are no boxes
    if len(indices) == 0:
        return []

    # filter detected objects
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # choose color for the label
        color = tuple(map(int, colors[label]))
        # draw box
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

        # draw label name inside the box
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 2000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame
from queue import Queue
frame_queue = Queue()

'''
run_object_detection
- source: Three video input modes are supported:
    - Video files:... /onvif/test/Road-vehicle-run.mp4
    - USB camera: 0 (depending on the value of the interface, it may be 0, or 1, or other)
    - RTSP stream: RTSP: / / 192.168.1.2 instead: 8080 / out h264
- flip: Some of the images from the camera are inverted. Here you need to flip.
-use_popup: Set this to True if we are running under.py and need to popup the video result, false if we are running in the notebook.
'''
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0, is_backed = False):
    global frame_queue
    player = None
    try:
        # create video player to play with target fps
        player = VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        # start capturing
        player.start()
        if use_popup and not is_backed:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            # grab the frame
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # if frame larger than full HD, reduce size to improve the performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # resize image and change dims to fit neural network input
            input_img = cv2.resize(
                src=frame, dsize=(height,width), interpolation=cv2.INTER_AREA
            )
            input_img = np.transpose(input_img, (2, 0, 1))  # 转置维度顺序

            # create batch of images (size = 1)
            input_img = input_img[np.newaxis, ...]

            # measure processing time

            start_time = time.time()
            # get results
            results = compiled_model([input_img])[output_layer]
            stop_time = time.time()
            # get poses from network results
            boxes = process_results(frame=frame, results=results)

            # draw boxes on a frame
            frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            # use processing times from last 200 frames
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # last_frame = frame
            frame_queue.put(frame, block=False)
            if is_backed:
                continue
            # use this workaround if there is flickering
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # encode numpy array to jpg
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # create IPython image
                i = display.Image(data=encoded_img)
                
                # display the image in this notebook
                display.clear_output(wait=True)
                display.display(i)
              
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # stop capturing
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # Video file example
    video_file = "tests/Road-vehicle-run.mp4"
    
    run_object_detection(source=video_file, flip=False, use_popup=True)
    # run_object_detection(source=video_rtsp, flip=False, use_popup=True)
    # run_object_detection(source=0, flip=True, use_popup=True)
