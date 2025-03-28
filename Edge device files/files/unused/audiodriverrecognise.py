import audiodriver
from pymongo import MongoClient
import takephoto
import cv2
import tensorflow as tf
from cameradriver import get_embedding 
import os
from videostream import VideoStream
import audiodriver

input("press enter")
audiodriver.embed_voice(3)
audiodriver.stop_listening()
