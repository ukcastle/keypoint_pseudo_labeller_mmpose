import cv2
import json
from pathlib import Path, PureWindowsPath
import shutil
SHOWNAME = "main"
PAD = 50
ZOOMRANGE= 300
NAME = ["hand", "shaft", "club_head"]
VISDICT = {0:"not exist", 1:"invisible", 2:"visible"}

import re
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
  return (*(arr[:,:2].min(0)), *(arr[:,2:].max(0)))

def drawBbox(img, imgW, imgH, imgPath):
  x1,y1,x2,y2 = getFullBbox(xywh2xyxy4Dict(getLabel(changePath(imgPath)), imgW, imgH))
  cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),1)

class StoreList(list):
  colorMap = ((0,255,0), (0,255,255), (0,0,255))
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.done = False
    self.cIdx = 0
    self.imgBuffer = []
    self.vis = 2
  def clickLeft(self,x,y,img):
    if len(self) >= 3:
      self.done = True
      return
    dst = img.copy()
    cv2.circle(dst, (x,y), radius=2, color=self.getColor(),thickness=cv2.FILLED)
    self.append((x,y,self.vis))
    self.cIdx = (self.cIdx+1) % 3
    self.imgBuffer.append(dst)
  def clickRight(self):
    if len(self) > 0:
      self.cIdx = (self.cIdx-1) % 3
      self.imgBuffer.pop()
      self.pop() 
  def getColor(self):
    return self.colorMap[self.cIdx]
  def makeDictFromList(self):
    elementDict = {}
    for i, v in enumerate(self):
      x,y,vis = v
      elementDict[NAME[i]] = {"x":x, "y":y, "vis":vis}
    return elementDict
  def notExistPoint(self):
    if len(self) >= 3:
      self.done = True
      return
    self.append((0,0,0))
    if len(self.imgBuffer) >= 1:
      self.imgBuffer.append(self.imgBuffer[-1])
class JsonWriter:

  def __init__(self, path) -> None:
    self.filePath = Path(path)
    self.outputDict = {}
    self.loadFile()
  
  def loadFile(self):
    if self.filePath.is_file():
      with self.filePath.open("r") as f:
        self.outputDict = json.load(f)  

  def addElement(self, fileName:str , elementDict):

    self.outputDict[fileName] = elementDict

  def saveFile(self):
    with self.filePath.open("w") as f:
      json.dump(self.outputDict, f, indent=2)
  
  @staticmethod
  def moveFile(path: Path, rootDir = "done"):
    donePath = Path(rootDir,*(path.parts[1:]))
    donePath.parent.mkdir(exist_ok=True, parents=True)
    shutil.move(path, donePath)
def drawCrossLine(img, x, y, crossLineHalf=10, color=(0,0,255)):
  cv2.line(img, (x-crossLineHalf,y), (x+crossLineHalf,y), color)
  cv2.line(img, (x,y-crossLineHalf), (x,y+crossLineHalf), color)
  return img
def getCropMatFromPoint(img, x,y,pad,imgW,imgH):
  x1, padx1 = (x-pad, 0) if (x-pad >= 0) else (0, abs(x-pad))
  y1, pady1 = (y-pad, 0) if (y-pad >= 0) else (0, abs(y-pad))
  x2, padx2 = (x+pad, 0) if (x+pad < imgW) else (imgW, abs(imgW-pad-x))
  y2, pady2 = (y+pad, 0) if (y+pad < imgH) else (imgH, abs(imgH-pad-y))
  return cv2.copyMakeBorder(img[y1:y2, x1:x2], pady1, pady2, padx1, padx2, cv2.BORDER_CONSTANT)
def mouseEvent(event, x, y, flags, param):
  img, imgW, imgH, sList = param
  img = sList.imgBuffer[-1] if len(sList.imgBuffer) > 0 else img

  if x >= imgW or y >= imgH:
    return
  
  div3 = int((imgH-ZOOMRANGE)/3)

  dst = img.copy()
  
  cv2.putText(dst, f"{VISDICT[sList.vis]}", (imgW-75, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
  for i,v in enumerate(sList): # 현재 찍힌 좌표값 미리보기 뿌려주기
    storeX, storeY, vis = v
    cropMat = getCropMatFromPoint(img, storeX, storeY, int(div3/4), imgW, imgH)
    dst[ZOOMRANGE+div3*(i):ZOOMRANGE+div3*(i+1), imgW:imgW+div3] = cv2.resize(cropMat, (int(div3), int(div3)))
    cv2.putText(dst, f"{NAME[i]} ({storeX},{storeY} {VISDICT[vis]})", (imgW+div3+10, int(ZOOMRANGE+div3*(i)+div3/2)), cv2.FONT_HERSHEY_PLAIN, 1, sList.colorMap[i])

  if event == cv2.EVENT_MOUSEMOVE: # 줌, 십자선 뿌려주기
    cropMat = getCropMatFromPoint(drawCrossLine(dst, x, y, color = sList.getColor()), x, y, PAD, imgW, imgH)
    dst[0:ZOOMRANGE, imgW:imgW+ZOOMRANGE] = cv2.resize(cropMat, (ZOOMRANGE,ZOOMRANGE)) #우측 위에 확대이미지
    
    cv2.line(dst, (imgW,ZOOMRANGE), (imgW+ZOOMRANGE, ZOOMRANGE), (255,255,255), 1) # 구분선

    cv2.line(dst, (0,y), (imgW,y), sList.getColor(), 1) #십자선 가로
    cv2.line(dst, (x,0), (x,imgH), sList.getColor(), 1) #십자선 세로
    cv2.imshow(SHOWNAME, dst)

  if event == cv2.EVENT_LBUTTONUP: 
    sList.clickLeft(x,y,img) # 포인트 추가
  if event == cv2.EVENT_RBUTTONUP:
    sList.clickRight() #포인트 삭제
    
def processOneImage(imgPath, jsonWriter : JsonWriter):
  img = cv2.imread(imgPath.as_posix())
  imgH, imgW = img.shape[:2]
  drawBbox(img, imgW, imgH, imgPath)
  img = cv2.copyMakeBorder(img, 0, 0, 0, ZOOMRANGE, cv2.BORDER_CONSTANT) # 정보창 공간 만들기
  sList = StoreList()
  cv2.imshow(SHOWNAME,img)
  cv2.setMouseCallback(SHOWNAME, mouseEvent, (img,imgW,imgH,sList))
  while(True):
    key = cv2.waitKey(1)
    if sList.done:
      jsonWriter.addElement(PureWindowsPath(*(imgPath.parts[1:])).as_posix(), sList.makeDictFromList())
      JsonWriter.moveFile(imgPath)
      break
    elif key== ord("`"):
      sList.notExistPoint()
    elif key== ord("1"):
      sList.vis = 1
    elif key== ord("2"):
      sList.vis = 2
    elif key== ord("r"):
      JsonWriter.moveFile(imgPath, "pass")
      break
    elif key==27: # esc
      exit(1)

if __name__=="__main__":
  try:
    j = JsonWriter("output.json")
    for p in Path("input\\outdoor_pro_best").rglob("**/*.jpg"):
      print(f"Current Num of Label : {j.outputDict.__len__()}")
      processOneImage(p, j)
  finally:
    j.saveFile()  
    