from math import sqrt
from pathlib import Path
VISDICT = {0:"not exist", 1:"invisible", 2:"visible"}
class ImagePointer:
  def __init__(self, outputList, bbox, imgName) -> None:
    self.imgName = imgName
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
  
  def changeVis(self):
    self.vis = (self.vis + 2) % 3
    if self.curSelectIdx:
      self.pointList[self.curSelectIdx][2] = self.vis

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

  def getHistoryTxt(self,fullLength):
    txtList = [""] * fullLength
    txtList[0] = f"vis : {self.vis} ({VISDICT[self.vis]})" 
    historyLen = len(self.history)
    for i in range(fullLength-1):
      if historyLen-1 < i:
        continue
      txtList[i+1] = str(self.history[-i-1])
    return txtList
  def __call__(self):
    return self.pointList

  def __repr__(self) -> str:
    return self.pointList.__repr__()
  

class ImagePointerList(list):
  def __init__(self, rootPath : Path, findGlob = "**/*.jpg", *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.iter = rootPath.rglob(findGlob)
    self.curIdx = -1
    self.metaDict = {}

  def next(self):
    self.curIdx += 1
    if self.curIdx < len(self):
      nextPath = self[self.curIdx]  
    else: 
      nextPath = next(self.iter)
      self.append(nextPath)
    return nextPath, self.metaDict[nextPath.as_posix()] if (nextPath.as_posix() in self.metaDict.keys()) else None
  
  def back(self):
    self.curIdx = self.curIdx - 1 if self.curIdx > 0 else 0
    backPath = self[self.curIdx]
    return backPath, self.metaDict[backPath.as_posix()] if (backPath.as_posix() in self.metaDict.keys()) else None
  
  def updateDict(self, curPath: Path, imagePointer, imgW, imgH):
    self.metaDict[curPath.as_posix()] = {
      "imagePointer" : imagePointer,
      "imgWH" : (imgW, imgH)
    }