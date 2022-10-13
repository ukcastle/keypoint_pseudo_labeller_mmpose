from time import time
import cv2
from pathlib import Path, PurePosixPath

from src.data_reader import drawBbox, moveFile
from src.img_handler import *
from src.model_helper import ModelHelper, KEYPOINTS
from src.point import ImagePointer, ImagePointerDict
from src.coco_writer import COCO_dict
from src.event_handler import EventHandler
from src.json_reader import *
SHOWNAME = "main"
PAD = 50
ZOOMRANGE = 300
INFORANGE = 200

HISTORY_SHOW_LEGNTH = 9
CONFIG = "src/model/golf_mobilenetv2_256x192.py"
WEIGHT = "src/model/best.pth"
SKELETONS = getSkeletons()
MODELWIDTH = 192
MODELHEIGHT = 256
DEVICE = "cuda:1"

def oneImageProcess(modelHelper, imgPath, value, preLabel):
  img = cv2.imread(imgPath.as_posix())
  if value is None:
    imgH,imgW = img.shape[:2]
    bbox = drawBbox(img, imgW, imgH, imgPath)
    x1,y1,x2,y2 = addPadding(*bbox, padding=0, imgW=imgW, imgH=imgH)
    output = modelHelper.inferenceModel(img, (x1,y1,x2,y2))
    imagePointer = ImagePointer(output, bbox, imgPath)
  else:
    imgW, imgH = value["imgWH"]
    imagePointer = value["imagePointer"]
    bbox = drawBbox(img, imgW, imgH, imgPath)

  parseOutputList(imagePointer, preLabel)

  img = initStatus(img, imgW, imgH, ZOOMRANGE, INFORANGE)
  cv2.imshow(SHOWNAME,drawSkeleton(img, imagePointer(), SKELETONS))
  
  eventHandler = EventHandler(imagePointer, imgW, imgH)
  
  cv2.setMouseCallback(SHOWNAME, eventHandler.mouseEvent)
  
  xy = eventHandler.xy

  startTime = time()
  while(True):
    outputMat = img.copy()
    key = cv2.waitKey(10)

    if (retVal := eventHandler.applyKeys(key)) is not None:
      if (time()-startTime) <= 1:
        continue
      return retVal

    putTextHistoryList(outputMat, HISTORY_SHOW_LEGNTH, 
      imagePointer.getHistoryTxt(HISTORY_SHOW_LEGNTH, eventHandler.viewLevel),
      imgW, imgH, ZOOMRANGE)

    putTextOutputInfo(outputMat, imagePointer, 
      KEYPOINTS, imgW+ZOOMRANGE, int(imgH/15))
    
    outputMat = drawKeyPointDot(outputMat, imagePointer())
    noDrawMat = outputMat.copy()
    
    outputMat = drawSkeleton(outputMat, imagePointer(), SKELETONS) 
    outputMat = drawKeyPointCircle(outputMat, imagePointer(), 5)

    noDrawMat = drawSkeleton(noDrawMat, imagePointer(), SKELETONS, viewLevel=eventHandler.viewLevel, curIdx=imagePointer.curSelectIdx) 
    
    if imagePointer.nowClicked and imagePointer.curSelectIdx is not None:
      cv2.putText(outputMat, f"{KEYPOINTS[imagePointer.curSelectIdx]}", (xy.x - 30, xy.y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
      cv2.putText(noDrawMat, f"{KEYPOINTS[imagePointer.curSelectIdx]}", (xy.x - 30, xy.y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
    
    lineColor = getRedColorByVis(imagePointer.vis) if not xy.isNextChange else (255,0,0)
    cropMat = getCropMatFromPoint(drawCrossLine(noDrawMat, xy.x, xy.y, color = lineColor), xy.x, xy.y, PAD, imgW, imgH)
    outputMat[0:ZOOMRANGE, imgW:imgW+ZOOMRANGE] = cv2.resize(cropMat, (ZOOMRANGE,ZOOMRANGE)) #우측 위에 확대이미지
    outputMat = drawFullCrossLine(outputMat, xy.x, xy.y, imgW, imgH, lineColor)

    cv2.imshow(SHOWNAME ,outputMat)

COCOJSON, BBOXJSON = "val.json", "valbbox.json"
ROOT = Path("data/input/outdoor_amateur_normal") # valid 7492

# COCOJSON, BBOXJSON = "coco.json", "bbox.json"
# ROOT = Path("data/input/outdoor_pro_best") # 9601

def checkHaveLabel(preLabel):
  for v in preLabel.values():
    if v is not None:
      return True
  return False

checkCnt = False

def main():
  modelHelper = ModelHelper(CONFIG,WEIGHT, modelWidth=MODELWIDTH, modelHeight=MODELHEIGHT, device=DEVICE)
  cocoDict = COCO_dict(COCOJSON, BBOXJSON)
  imgStorage = ImagePointerDict(ROOT)
  cnt=0
  try:
    isNext = True
    while(True):
      nextPath, value = imgStorage.next() if isNext else imgStorage.back()
      preLabel = getLabelFromJson(getJsonPathByImgPath(nextPath))
      
      if checkHaveLabel(preLabel):
        cnt+=1
        if checkCnt:
          continue    
        isNext, processRetVal = oneImageProcess(modelHelper, nextPath, value, preLabel)

        if processRetVal is None:
          imgStorage.passIdx(nextPath)
          continue

        print(imgStorage.curIdx, cnt, nextPath)
        imagePointer, imgW, imgH = processRetVal
        imgStorage.updateDict(imagePointer.imgName , imagePointer, imgW, imgH)
  except StopIteration as e:
    return

  finally:
    if checkCnt:
      print(cnt)
      exit(0)
    for key, val in imgStorage.items():
      if val is None:
        moveFile(key, rootDir="data/pass")
        continue
      imgPath = PurePosixPath(*(key.split("/")[2:]))
      cocoDict.updateDict(imgPath, val["imagePointer"].pointList, val["imagePointer"].bbox, *val["imgWH"])
      moveFile(key)
    # coco 저장
    cocoDict.saveCOCO()
    cocoDict.saveBbox()
    pass
if __name__=="__main__":
  main()