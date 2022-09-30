import cv2
from src.util import timeit

def initStatus(img, imgW, imgH, zoomRange, infoRange):
  img = cv2.copyMakeBorder(img, 0, 0, 0, zoomRange+infoRange, cv2.BORDER_CONSTANT) # 정보창
  cv2.line(img, (imgW,zoomRange), (imgW+zoomRange, zoomRange), (255,255,255), 2) # 구분선
  cv2.line(img, (imgW+zoomRange,0), (imgW+zoomRange,imgH), (255,255,255), 2)
  return img
def drawCrossLine(img, x, y, crossLineHalf=10, color=(0,0,255)):
  cv2.line(img, (x-crossLineHalf,y), (x+crossLineHalf,y), color)
  cv2.line(img, (x,y-crossLineHalf), (x,y+crossLineHalf), color)
  return img

def putTextHistoryList(outputMat, historyLength, txtList, imgW, imgH, zoomRange):
  historyTabDiv4 = int((imgH-zoomRange) / historyLength)
  for i in range(historyLength):
    color = (0,0,255) if i==1 else (255,255,255)
    cv2.putText(outputMat, txtList[i], 
      (imgW, zoomRange+historyTabDiv4*(i+1)-int(historyTabDiv4/historyLength)), 
      cv2.FONT_HERSHEY_PLAIN, 1, color)

def putTextOutputInfo(outputMat, imagePointer, keyPoints, width, height):
  for i in range(15):
    color = (0,0,255) if i==imagePointer.curSelectIdx else (255,255,255)
    cv2.putText(outputMat, f"{keyPoints[i]}", (width+10, height*(i)+15), 
      cv2.FONT_HERSHEY_PLAIN, 1, color)
    cv2.putText(outputMat, f"{imagePointer.predTxt[i]}", (width+130, height*(i)+15), 
      cv2.FONT_HERSHEY_PLAIN, 1, color)
    cv2.putText(outputMat, f"{imagePointer()[i]}", (width+20, height*(i)+30), 
      cv2.FONT_HERSHEY_PLAIN, 1, color)

def getCropMatFromPoint(img, x,y,pad,imgW,imgH):
  x1, padx1 = (x-pad, 0) if (x-pad >= 0) else (0, abs(x-pad))
  y1, pady1 = (y-pad, 0) if (y-pad >= 0) else (0, abs(y-pad))
  x2, padx2 = (x+pad, 0) if (x+pad < imgW) else (imgW, abs(imgW-pad-x))
  y2, pady2 = (y+pad, 0) if (y+pad < imgH) else (imgH, abs(imgH-pad-y))
  return cv2.copyMakeBorder(img[y1:y2, x1:x2], pady1, pady2, padx1, padx2, cv2.BORDER_CONSTANT)

def drawFullCrossLine(img, x, y, imgW, imgH, color):
  cv2.line(img, (x,0), (x, imgH), color)
  cv2.line(img, (0,y), (imgW, y), color)
  return img

def addPadding(x1, y1, x2, y2, padding, imgW, imgH):
  x1, y1 = (max(p-padding, 0) for p in (x1, y1))
  x2, y2 = (min(p+padding, length) for p, length in ((x2,imgW), (y2, imgH)))
  return x1, y1, x2 ,y2

def getKeyDict():
  keyDict = {}
  for value in dataset_info["keypoint_info"].values():
    keyDict[value["name"]] = value["id"]
  return keyDict

def getSkeletons():
  keyDict = getKeyDict()
  skeletonList = []
  for value in dataset_info["skeleton_info"].values():
    startId, endId = [keyDict[x] for x in value["link"]] 
    skeletonList.append((startId, endId, value["color"]))

  return skeletonList

from .model.custom_golf import dataset_info
# @timeit
def drawKeyPointCircle(img, outputList, radius):
  dst = img.copy()
  for i,v in enumerate(outputList):
    x,y,vis = [int(item) for item in v]
    keyDict = dataset_info["keypoint_info"][i]
    cv2.circle(dst, (x,y), radius, keyDict["color"])
    
  return dst

def drawKeyPointDot(img, outputList, radius = 1):
  dst = img.copy()
  for i,v in enumerate(outputList):
    x,y,vis = [int(item) for item in v]
    cv2.circle(dst, (x,y), radius, getRedColorByVis(vis))
  
  return dst
# @timeit
def drawSkeleton(img, outputList, skeletons, viewLevel=2, curIdx = None):
  dst = img.copy()
  for startIdx, endIdx, color in skeletons:
    if viewLevel == 0:
      break
    if (viewLevel == 1) and (curIdx not in (startIdx, endIdx)):
      continue
    start = outputList[startIdx][:2]
    end = outputList[endIdx][:2]
    cv2.line(dst, start, end, color)
  return dst

def getRedColorByVis(vis):
  return (127*(2-vis),127*(2-vis),255)