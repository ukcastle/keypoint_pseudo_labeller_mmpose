import cv2
from pathlib import Path, PurePosixPath

from src.data_reader import drawBbox, moveFile
from src.img_handler import *
from src.model_helper import ModelHelper, KEYPOINTS
from src.point import ImagePointer, ImagePointerDict
from src.coco_writer import COCO_dict
from src.event_handler import EventHandler

SHOWNAME = "main"
PAD = 50
ZOOMRANGE = 300
INFORANGE = 200
HISTORY_SHOW_LEGNTH = 9
CONFIG = "src/model/golf_mobilenetv2_256x192.py"
WEIGHT = "src/model/best.pth"
SKELETONS = getSkeletons()
VIS_THRESH = 0.6
MODELWIDTH = 192
MODELHEIGHT = 256
DEVICE = "cuda:1"



def oneImageProcess(modelHelper, imgPath, value):
  img = cv2.imread(imgPath.as_posix())
  if value is None:
    imgH,imgW = img.shape[:2]
    bbox = drawBbox(img, imgW, imgH, imgPath)
    x1,y1,x2,y2 = addPadding(*bbox, padding=30, imgW=imgW, imgH=imgH)
    output = modelHelper.inferenceModel(img, (x1,y1,x2,y2))
    imagePointer = ImagePointer(output, bbox, imgPath)
  else:
    imgW, imgH = value["imgWH"]
    imagePointer = value["imagePointer"]
    bbox = drawBbox(img, imgW, imgH, imgPath)

  img = cv2.copyMakeBorder(img, 0, 0, 0, ZOOMRANGE+INFORANGE, cv2.BORDER_CONSTANT) # 정보창
  cv2.line(img, (imgW,ZOOMRANGE), (imgW+ZOOMRANGE, ZOOMRANGE), (255,255,255), 2) # 구분선
  cv2.line(img, (imgW+ZOOMRANGE,0), (imgW+ZOOMRANGE,imgH), (255,255,255), 2)
  cv2.imshow(SHOWNAME,drawSkeleton(img, imagePointer(), SKELETONS))
  
  eventHandler = EventHandler(imagePointer, imgW, imgH)
  
  cv2.setMouseCallback(SHOWNAME, eventHandler.mouseEvent)
  historyTabDiv4 = int((imgH-ZOOMRANGE) / HISTORY_SHOW_LEGNTH)
  outputInfoDiv15 = int(imgH/15)
  
  xy = eventHandler.xy

  while(True):
    outputMat = img.copy()
    key = cv2.waitKey(10)

    if retVal := (eventHandler.applyKeys(key)) is not None:
      return retVal

    txtList = imagePointer.getHistoryTxt(HISTORY_SHOW_LEGNTH)

    for i in range(HISTORY_SHOW_LEGNTH):
      color = (0,0,255) if i==1 else (255,255,255)
      cv2.putText(outputMat, txtList[i], 
        (imgW, ZOOMRANGE+historyTabDiv4*(i+1)-int(historyTabDiv4/HISTORY_SHOW_LEGNTH)), 
        cv2.FONT_HERSHEY_PLAIN, 1, color)
    
    for i in range(15):
      color = (0,0,255) if i==imagePointer.curSelectIdx else (255,255,255)
      cv2.putText(outputMat, f"{KEYPOINTS[i]}", (imgW+ZOOMRANGE+10, outputInfoDiv15*(i)+15), 
        cv2.FONT_HERSHEY_PLAIN, 1, color)
      cv2.putText(outputMat, f"{imagePointer.predTxt[i]}", (imgW+ZOOMRANGE+130, outputInfoDiv15*(i)+15), 
        cv2.FONT_HERSHEY_PLAIN, 1, color)
      cv2.putText(outputMat, f"{imagePointer()[i]}", (imgW+ZOOMRANGE+20, outputInfoDiv15*(i)+30), 
        cv2.FONT_HERSHEY_PLAIN, 1, color)

    outputMat = drawKeyPointDot(outputMat, imagePointer())
    noDrawMat = outputMat.copy()
    
    outputMat = drawSkeleton(outputMat, imagePointer(), SKELETONS) 
    outputMat = drawKeyPointCircle(outputMat, imagePointer(), 5)
    if eventHandler.isShowSkeletonInCrop:
      noDrawMat = drawSkeleton(noDrawMat, imagePointer(), SKELETONS, viewLevel=1, curIdx=imagePointer.curSelectIdx) 
    
    if imagePointer.nowClicked and imagePointer.curSelectIdx is not None:
      cv2.putText(outputMat, f"{KEYPOINTS[imagePointer.curSelectIdx]}", (xy.x - 30, xy.y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
      cv2.putText(noDrawMat, f"{KEYPOINTS[imagePointer.curSelectIdx]}", (xy.x - 30, xy.y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255))
    
    lineColor = getRedColorByVis(imagePointer.vis) if not xy.isNextChange else (255,0,0)
    cropMat = getCropMatFromPoint(drawCrossLine(noDrawMat, xy.x, xy.y, color = lineColor), xy.x, xy.y, PAD, imgW, imgH)
    outputMat[0:ZOOMRANGE, imgW:imgW+ZOOMRANGE] = cv2.resize(cropMat, (ZOOMRANGE,ZOOMRANGE)) #우측 위에 확대이미지
    outputMat = drawFullCrossLine(outputMat, xy.x, xy.y, imgW, imgH, lineColor)

    cv2.imshow(SHOWNAME ,outputMat)

ROOT = Path("data/input/outdoor_amateur_")
def main():
  modelHelper = ModelHelper(CONFIG,WEIGHT, modelWidth=MODELWIDTH, modelHeight=MODELHEIGHT, device=DEVICE)
  cocoDict = COCO_dict("coco.json", "bbox.json")
  # cocoDict = COCO_dict("val.json", "valbbox.json")
  imgStorage = ImagePointerDict(ROOT)
  try:
    isNext = True
    while(True):
      nextPath, value = imgStorage.next() if isNext else imgStorage.back()
      isNext, value = oneImageProcess(modelHelper, nextPath, value)

      if value is None:
        imgStorage.passIdx(nextPath)
        continue

      print(imgStorage.curIdx, nextPath)
      imagePointer, imgW, imgH = value
      imgStorage.updateDict(imagePointer.imgName , imagePointer, imgW, imgH)
         
  finally:
    # for key, val in imgStorage.items():
    #   if val is None:
    #     moveFile(key, rootDir="data/pass")
    #     continue
    #   imgPath = PurePosixPath(*(key.split("/")[2:]))
    #   cocoDict.updateDict(imgPath, val["imagePointer"].pointList, val["imagePointer"].bbox, *val["imgWH"])
    #   moveFile(key)
    # # coco 저장
    # cocoDict.saveCOCO()
    # cocoDict.saveBbox()
    pass
if __name__=="__main__":
  main()