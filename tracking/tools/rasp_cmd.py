import subprocess
import time

subprocess.Popen('raspivid -vs  -vf -n -w 1280 -h 720 -b 25000000 -fps 30 -vf -hf -t 0 -o - | gst-launch-1.0 -v fdsrc !  h264parse ! tee name=splitter ! queue ! rtph264pay config-interval=10 pt=96 ! udpsink host=192.168.0.100 port=9000 splitter. ! queue ! filesink location="videofile.h264"'
,shell=True)

while True:
    print('ya')
    time.sleep(1)
