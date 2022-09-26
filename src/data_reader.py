
from pathlib import Path
import re
import cv2

def getLabel(lbPath):
  retDict = {}
  with lbPath.open("r") as f:
    for finded in re.findall("(\d)\s(\d.\d{6})\s(\d.\d{6})\s(\d.\d{6})\s(\d.\d{6})",f.read()):
      retDict[int(finded[0])] = [float(x) for x in finded[1:]]
  return retDict

def changePath(path, basePath="images", dstPath="labels", dstSuffix=".txt"):
  idx = path.parts.index(basePath)
  return Path(*path.parts[:idx], dstPath, *path.parts[idx+1:]).with_suffix(dstSuffix)

def xywh2xyxy(xc, yc, w, h, imgW, imgH):
  x1, x2 = (int(imgW*x) for x in (xc-w/2, xc+w/2))
  y1, y2 = (int(imgH*y) for y in (yc-h/2, yc+h/2))
  return x1, y1, x2, y2

def xywh2xyxy4Dict(lbDict : dict, imgW, imgH):
  retDict = lbDict.copy()
  for key in lbDict.keys():
    retDict[key] = xywh2xyxy(*lbDict[key], imgW, imgH)  
  return retDict

import numpy as np
def getFullBbox(labels : dict):
  arr = np.array(list(labels.values()))
  return [int(x) for x in (*(arr[:,:2].min(0)), *(arr[:,2:].max(0)))]

def drawBbox(img, imgW, imgH, imgPath):
  x1,y1,x2,y2 = getFullBbox(xywh2xyxy4Dict(getLabel(changePath(imgPath)), imgW, imgH))
  cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),1)
  return x1,y1,x2,y2