from math import sqrt
import cv2
from pathlib import Path
SHOWNAME = "main"
PAD = 50
ZOOMRANGE= 300
VISDICT = {0:"not exist", 1:"invisible", 2:"visible"}

from src.data_reader import drawBbox
from src.img_handler import *
from pathlib import Path
from src.model_helper import ModelHelper
CONFIG = "src/model/golf_mobilenetv2_256x192.py"
WEIGHT = "src/model/latest.pth"
MODELWIDTH = 192
MODELHEIGHT = 256
DEVICE = "cuda:1"
SKELETONS = getSkeletons()
VIS_THRESH = 0.6
KEYMAP = {ord(str(x)) : i for i, x in enumerate([*range(1,10),0, "q", "w", "e", "r", "t"])}
  

class PointManager:
  def __init__(self, outputList, bbox) -> None:
    self.curSelectIdx = None
    self.pointList = []
    self.history = []
    self.nowClicked = False
    self.vis = 2
    self.bbox = bbox
    for i,v in enumerate(outputList):
      x, y, p = v
      vis = 2
      if i < 11 and p < 0.6:
        vis = 1
      self.pointList.append([int(x),int(y),vis])

  def getNearIdx(self, x, y, thresh=10):
    for i,v in enumerate(self.pointList):
      pointX,pointY,_ = v 
      if sqrt((pointX-x)**2 + (pointY-y)**2) < thresh:
        return i
    return None
  
  def setPoint(self, x, y):
    if self.isNullSelect():
      return
    bx1, by1, bx2, by2 = self.bbox
    x = bx1 if x<bx1+1 else x 
    x = bx2 if x>bx2-1 else x 
    y = by1 if y<by1+1 else y 
    y = by2 if y>by2-1 else y 
    self.pointList[self.curSelectIdx] = [x,y,self.vis]

  def addHistory(self):
    self.history.append((self.curSelectIdx, self.pointList[self.curSelectIdx].copy()))

  def setSelected(self, i):
    self.curSelectIdx = i

  def isNullSelect(self):
    return self.curSelectIdx is None

  def rollback(self):
    if len(self.history) == 0:
      return
    idx, val = self.history.pop()
    x,y,self.vis = val
    self.setSelected(idx)
    self.setPoint(x,y)
    self.setSelected(None)

  def __call__(self):
    return self.pointList

  def __repr__(self) -> str:

    return self.pointList.__repr__()

from src.util import timeit

# @timeit
def mouseEvent(event, x, y, flags, param):
  
  img, imgW, imgH, pm = param
  dst = img.copy()
  
  # 클릭할 때 Null이면 nearidx 찾기
  # 선택이 완료됐으면 History에 남기기
  if event == cv2.EVENT_LBUTTONDOWN:
    pm.nowClicked = True
    if pm.isNullSelect():
      pm.setSelected(pm.getNearIdx(x, y))
    if not pm.isNullSelect():
      pm.addHistory()

  # 마우스 업일때 클릭된거 초기화해주기
  elif event == cv2.EVENT_LBUTTONUP:
    pm.nowClicked = False
    if not pm.isNullSelect(): 
      pm.setSelected(None)

  # 드래그할때 setpoint로 설정해주기
  elif event == cv2.EVENT_MOUSEMOVE:
    x = min(x, imgW)
    if pm.nowClicked:
      pm.setPoint(x,y)
    dst = drawSkeleton(dst, pm(), SKELETONS)

  elif event == cv2.EVENT_RBUTTONDOWN:
    pm.rollback()

  
  dst = drawSkeleton(dst, pm(), SKELETONS)
  cropMat = getCropMatFromPoint(drawCrossLine(dst, x, y, color = (0,0,255)), x, y, PAD, imgW, imgH)
  dst[0:ZOOMRANGE, imgW:imgW+ZOOMRANGE] = cv2.resize(cropMat, (ZOOMRANGE,ZOOMRANGE)) #우측 위에 확대이미지
  dst = drawFullCrossLine(dst, x, y, imgW, imgH, (0,0,255))
  cv2.imshow(SHOWNAME ,dst)

  

def main():
  modelHelper = ModelHelper(CONFIG,WEIGHT, modelWidth=MODELWIDTH, modelHeight=MODELHEIGHT, device=DEVICE)
  
  imgPath = Path("data\\input\\indor_semi_best\\images\\20201123_General_003_DIS_S_F20_SS_001_3832.jpg")
  img = cv2.imread(imgPath.as_posix())
  imgH,imgW = img.shape[:2]

  # Bbox 그리고 Inference용 패딩 추가 이미지 만들기
  bbox = drawBbox(img, imgW, imgH, imgPath)
  x1,y1,x2,y2 = addPadding(*bbox, padding=30, imgW=imgW, imgH=imgH)
  output = modelHelper.inferenceModel(img, (x1,y1,x2,y2))
  pm = PointManager(output, bbox)

  img = cv2.copyMakeBorder(img, 0, 0, 0, ZOOMRANGE, cv2.BORDER_CONSTANT) # 정보창
  cv2.line(img, (imgW,ZOOMRANGE), (imgW+ZOOMRANGE, ZOOMRANGE), (255,255,255), 2) # 구분선
  
  cv2.imshow(SHOWNAME,drawSkeleton(img, pm(), SKELETONS))
  cv2.setMouseCallback(SHOWNAME, mouseEvent, (img, imgW, imgH, pm))
  while(True):
    key = cv2.waitKey(1)
    if key in KEYMAP.keys():
      # 키보드로 인덱스 설정 대신 할수있게하기
      pm.setSelected(KEYMAP[key])

    elif key==27: # esc
      exit(1)

if __name__=="__main__":
  main()