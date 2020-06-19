from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import stream
import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob
import serial
import binascii

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import socket
import sys

# try:
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# except socket.error as msg:
#     sys.stderr.write("[ERROR] %s\n" % msg[1])
#     sys.exit(1)

# try:
#     sock.connect(('192.168.0.101', 9001))
# except socket.error as msg:
#     sys.stderr.write("[ERROR] %s\n" % msg[1])
#     exit(1)

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file',
                    default='pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml')
parser.add_argument('--snapshot', type=str, help='model name',
                    default='pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
print(ser.name)  # 列印裝置名稱
print(ser.port)  # 列印裝置名
if not ser.isOpen():
    ser.open()

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
x_size = 1080
y_size = 1080


def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0, 2]] = x
        drawnBox[[1, 3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0, 2]] = np.sort(boxToDraw[[0, 2]])
    boxToDraw[[1, 3]] = np.sort(boxToDraw[[1, 3]])


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    video = stream.Video()
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback(video_name, on_mouse, 0)

    # for frame in get_frames(args.video_name):
    while True:
        if not video.frame_available():
            continue
        frame = video.frame()

        if first_frame:
            # try:
            #     init_rect = cv2.selectROI(video_name, frame, False, False)
            # except:
            #     exit()
            # tracker.init(frame, init_rect)
            # first_frame = False
            if mousedown:
                cv2.rectangle(frame,
                              (int(boxToDraw[0]), int(boxToDraw[1])),
                              (int(boxToDraw[2]), int(boxToDraw[3])),
                              [0, 0, 255], 5)

            elif mouseupdown:
                cv2.rectangle(frame,
                              (int(boxToDraw[0]), int(boxToDraw[1])),
                              (int(boxToDraw[2]), int(boxToDraw[3])),
                              [0, 0, 255], 5)
                init_rect = boxToDraw
                init_rect[2] = boxToDraw[2]-boxToDraw[0]
                init_rect[3] = boxToDraw[3]-boxToDraw[1]
                tracker.init(frame, init_rect)
                first_frame = False
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
        else:

            if mousedown:
                first_frame = True
                continue
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)

                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                x_mid = bbox[0]+0.5*bbox[2]
                print(x_mid)
                y_mid = bbox[1]+0.5*bbox[3]
                print(y_mid)

                speed=30

                stringi = 'i='+str(round(x_mid-(0.5*x_size))*speed)+' '
                stringj = 'j='+str(round((0.5*y_size)-y_mid)*speed)+' '

                hexi_b = ''.join(hex(ord(c))[2:] for c in stringi)
                print(hexi_b)
                print(bytes.fromhex(hexi_b))
                hexj_b = ''.join(hex(ord(c))[2:] for c in stringj)
                print(hexj_b)
                print(bytes.fromhex(hexj_b))
                ser.write(bytes.fromhex(hexi_b))
                ser.write(bytes.fromhex(hexj_b))

                # ser.write(b'\x80\x47\x4F\x53\x55\x42\x33\x20')

                # sock.send(b([round(x_mid)])) 
                # sock.send(b([round(y_mid)]))
                # print(sock.recv(1024))

                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
