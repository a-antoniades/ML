#!/bin/sh
for i in {1..100}
do
    rm -f /tf_files/Demonstration*
	python3 captureFrame.py $i
	python -m scripts.label_image --graph=tf_files/retrained_graph.pb --image=tf_files/Demonstration/img$i.jpg
done
