from math import sqrt
from pathlib import Path
from .img_handler import getKeyDict
from .model.custom_golf import dataset_info
VISDICT = {0:"not exist", 1:"invisible", 2:"visible"}
KEYDICT = getKeyDict()

with open("src/define.yaml", "r") as f:
  import yaml
  KEYPOINTS = yaml.load(f, Loader=yaml.FullLoader)["keypoints"]
class ImagePointer:
  def __init__(self, outputList, bbox, imgName) -> None:
    self.imgName = imgName
    self.curSelectIdx = None
    self.pointList = []
    self.history = []
    self.predTxt = []
    self.nowClicked = False
    self.vis = 2
    self.bbox = bbox
    for i,v in enumerate(outputList):
      x, y, p = v
      vis = 2
      self.pointList.append([int(x),int(y),vis])
      self.predTxt.append(f"{p:.2f}")

  def getNearIdx(self, x, y, thresh=10):
    shortestIdx, shortestDistance = -1, 9999
    for i,v in enumerate(self.pointList):
      pointX,pointY,_ = v 
      curDistance = sqrt((pointX-x)**2 + (pointY-y)**2)
      if curDistance < shortestDistance:
        shortestIdx = i
        shortestDistance = curDistance
    if shortestDistance > thresh:
      return None
    else:
      return shortestIdx
  
  def changeVis(self, absVal = -1):
    if absVal in range(3):
      self.vis = absVal
    else:
      self.vis = (self.vis + 2) % 3
    if self.curSelectIdx:
      self.pointList[self.curSelectIdx][2] = self.vis

  def setPoint(self, x, y, vis = None, seletedIdx = None):
    if (seletedIdx is None) and self.isNullSelect():
      return
    bx1, by1, bx2, by2 = self.bbox
    x = bx1 if x<bx1+1 else x 
    x = bx2 if x>bx2-1 else x 
    y = by1 if y<by1+1 else y 
    y = by2 if y>by2-1 else y 
    
    curVis = self.vis if (vis is None) else vis
    curSeletedIdx = self.curSelectIdx if (seletedIdx is None) else seletedIdx 
    self.pointList[curSeletedIdx] = [x,y,curVis]
    
  def addHistory(self):
    self._addHistory(self.curSelectIdx)

  def _addHistory(self, curIdx):
    self.history.append((curIdx, self.pointList[curIdx].copy()))

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

  def getHistoryTxt(self,fullLength, skeletonVis):
    txtList = [""] * fullLength
    txtList[0] = f"vis : {self.vis} ({VISDICT[self.vis]}), viewLevel : {skeletonVis}" 
    historyLen = len(self.history)
    for i in range(fullLength-1):
      if historyLen-1 < i:
        continue
      idx, val = self.history[-i-1]
      txtList[i+1] = f"{KEYPOINTS[idx]}, {val}"
    return txtList

  def changePair(self):
    if self.isNullSelect():
      return
    if (swapIdx := KEYDICT[dataset_info["keypoint_info"][self.curSelectIdx]["swap"]]) == "":
      return
    self._addHistory(self.curSelectIdx)
    self._addHistory(swapIdx)
    temp = self.pointList[self.curSelectIdx].copy()
    self.pointList[self.curSelectIdx] = self.pointList[int(swapIdx)].copy()
    self.pointList[int(swapIdx)] = temp

  def __call__(self):
    return self.pointList

  def __repr__(self) -> str:
    return self.pointList.__repr__()
  

class ImagePointerDict(dict):
  def __init__(self, rootPath : Path, findGlob = "**/*.jpg", *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.iter = rootPath.rglob(findGlob)
    self.curIdx = -1
    self.pathList = []
    self.passIdxList = []

  def passIdx(self, curPath):
    self.passIdxList.append(self.curIdx)
    self[curPath.as_posix()] = None

  def next(self):
    self.curIdx += 1
    if self.curIdx < len(self):
      while (self.curIdx in self.passIdxList):
        self.curIdx+=1
      nextPath = self.pathList[self.curIdx]  
    else: 
      nextPath = next(self.iter)
      self.pathList.append(nextPath)
    return nextPath, self[nextPath.as_posix()] if (nextPath.as_posix() in self.keys()) else None
  
  def back(self):
    self.curIdx = max(self.curIdx - 1,0)
    isZeroPass = False
    while (self.curIdx in self.passIdxList):
      if self.curIdx <= 0:
        isZeroPass = True
      if isZeroPass:
        self.curIdx += 1
      else:
        self.curIdx -= 1
    backPath = self.pathList[self.curIdx]
    return backPath, self[backPath.as_posix()] if (backPath.as_posix() in self.keys()) else None
  
  def updateDict(self, curPath: Path, imagePointer, imgW, imgH):
    self[curPath.as_posix()] = {
      "imagePointer" : imagePointer,
      "imgWH" : (imgW, imgH)
    }