import cv2
from pathlib import Path, PureWindowsPath
from src.data_reader import drawBbox
from src.img_handler import *
from src.model_helper import ModelHelper, KEYPOINTS
from src.point import ImagePointer, ImagePointerList

class X_Y:
  def __init__(self) -> None:
    self.x=0
    self.y=0

SHOWNAME = "main"
PAD = 50
ZOOMRANGE = 300
INFORANGE = 200
HISTORY_SHOW_LEGNTH = 9
CONFIG = "src/model/golf_mobilenetv2_256x192.py"
WEIGHT = "src/model/latest.pth"
MODELWIDTH = 192
MODELHEIGHT = 256
DEVICE = "cuda:1"
SKELETONS = getSkeletons()
VIS_THRESH = 0.6
KEYMAP = {ord(str(x)) : i for i, x in enumerate([*range(1,10),0, "q", "w", "e", "r", "t"])}


def oneImageProcess(modelHelper, imageList, strPath):
  imgPath = Path(strPath)
  img = cv2.imread(imgPath.as_posix())
  imgH,imgW = img.shape[:2]

  bbox = drawBbox(img, imgW, imgH, imgPath)
  x1,y1,x2,y2 = addPadding(*bbox, padding=30, imgW=imgW, imgH=imgH)
  output = modelHelper.inferenceModel(img, (x1,y1,x2,y2))

  imagePointer = ImagePointer(output, bbox, PureWindowsPath(*(imgPath.parts[1:])))
  img = cv2.copyMakeBorder(img, 0, 0, 0, ZOOMRANGE+INFORANGE, cv2.BORDER_CONSTANT) # 정보창
  cv2.line(img, (imgW,ZOOMRANGE), (imgW+ZOOMRANGE, ZOOMRANGE), (255,255,255), 2) # 구분선
  cv2.line(img, (imgW+ZOOMRANGE,0), (imgW+ZOOMRANGE,imgH), (255,255,255), 2)
  cv2.imshow(SHOWNAME,drawSkeleton(img, imagePointer(), SKELETONS))
  xy = X_Y()
  cv2.setMouseCallback(SHOWNAME, mouseEvent, (imgW, imgH, imagePointer, xy))
  historyTabDiv4 = int((imgH-ZOOMRANGE) / HISTORY_SHOW_LEGNTH)
  outputInfoDiv15 = int(imgH/15)
  while(True):
    outputMat = img.copy()
    key = cv2.waitKey(1)
    if key in KEYMAP.keys():
      # 키보드로 인덱스 설정 대신 할수있게하기
      imagePointer.setSelected(KEYMAP[key])
    elif key==ord("a"):
      imagePointer.changeVis()
    elif key==ord("s"):
      imageList.append(imagePointer)
      break
    elif key==27: # esc
      break

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
      cv2.putText(outputMat, f"{imagePointer()[i]}", (imgW+ZOOMRANGE+20, outputInfoDiv15*(i)+30), 
        cv2.FONT_HERSHEY_PLAIN, 1, color)

    outputMat = drawSkeleton(outputMat, imagePointer(), SKELETONS)
    outputMat = drawKeyPointCircle(outputMat, imagePointer(), 5)
    cropMat = getCropMatFromPoint(drawCrossLine(outputMat, xy.x, xy.y, color = (0,0,255)), xy.x, xy.y, PAD, imgW, imgH)
    outputMat[0:ZOOMRANGE, imgW:imgW+ZOOMRANGE] = cv2.resize(cropMat, (ZOOMRANGE,ZOOMRANGE)) #우측 위에 확대이미지
    outputMat = drawFullCrossLine(outputMat, xy.x, xy.y, imgW, imgH, (0,0,255))

    cv2.imshow(SHOWNAME ,outputMat)

def mouseEvent(event, x, y, flags, param):
  
  imgW, imgH, imagePointer, xy = param
  xy.x, xy.y = x, y
  
  # 클릭할 때 Null이면 nearidx 찾기
  # 선택이 완료됐으면 History에 남기기
  if event == cv2.EVENT_LBUTTONDOWN:
    imagePointer.nowClicked = True
    if imagePointer.isNullSelect():
      imagePointer.setSelected(imagePointer.getNearIdx(x, y))
    if not imagePointer.isNullSelect():
      imagePointer.addHistory()

  # 마우스 업일때 클릭된거 초기화해주기
  elif event == cv2.EVENT_LBUTTONUP:
    imagePointer.nowClicked = False
    if not imagePointer.isNullSelect(): 
      imagePointer.setSelected(None)

  # 드래그할때 setpoint로 설정해주기
  elif event == cv2.EVENT_MOUSEMOVE:
    x = min(x, imgW)
    if imagePointer.nowClicked:
      imagePointer.setPoint(x,y)

  elif event == cv2.EVENT_RBUTTONDOWN:
    imagePointer.rollback()

def main():
  modelHelper = ModelHelper(CONFIG,WEIGHT, modelWidth=MODELWIDTH, modelHeight=MODELHEIGHT, device=DEVICE)
  imageList = ImagePointerList()
  oneImageProcess(modelHelper, imageList, "data\\input\\indor_semi_best\\images\\20201123_General_003_DIS_S_F20_SS_001_3832.jpg")
  print(imageList.makeDictFromList())

  # Bbox 그리고 Inference용 패딩 추가 이미지 만들기
  
 
if __name__=="__main__":
  main()