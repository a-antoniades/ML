import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture(0)
success,frame = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file
  success,frame = vidcap.read()
  print 'Read a new frame: ', success
  count += 1
      -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/Testing/fish.jpge\