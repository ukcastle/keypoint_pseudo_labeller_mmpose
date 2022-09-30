import cv2
from .point import ImagePointer
class X_Y:
  def __init__(self, imgW, imgH) -> None:
    self.x=0
    self.y=0
    self.isNextChange=False
    self.imgW = imgW
    self.imgH = imgH

  def setXY(self, x, y):
    self.x = min(x,self.imgW)
    self.y = min(y,self.imgH)

KEYMAP = {ord(str(x)) : i for i, x in enumerate([*range(1,10),0, "q", "w", "e", "r", "t"])}

class EventHandler:
  viewLevel = 1 #static
  def __init__(self, imagePointer : ImagePointer, imgW, imgH) -> None:
    self.xy = X_Y(imgW, imgH)
    self.imagePointer = imagePointer
    self.imgW = imgW
    self.imgH = imgH
    
  def applyKeys(self, key):
    if key in KEYMAP.keys():
      # 키보드로 인덱스 설정 대신 할수있게하기
      self.imagePointer.setSelected(KEYMAP[key])
    elif key==32: # space
      return True, (self.imagePointer, self.imgW, self.imgH)
    elif key==ord("b"):
      return False, (self.imagePointer, self.imgW, self.imgH)
    elif key==ord("a"):
      self.imagePointer.changeVis()
    elif key==ord("s"):
      self.imagePointer.changeVis(absVal=2)
    elif key==ord("p"):
      return True, None
    elif key==ord("h") or key==ord("v"):
      EventHandler.viewLevel = (EventHandler.viewLevel+1) % 3
    elif key==ord("c"):
      self.xy.isNextChange = not self.xy.isNextChange
    elif key==27: # esc
      exit()
    
    return None


  def mouseEvent(self, event, x, y, flags, param):
    self.xy.setXY(x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
      self._lBtnDown(x, y)  

    elif event == cv2.EVENT_LBUTTONUP:
      self._lBtnUp()

    # 드래그할때 setpoint로 설정해주기
    elif event == cv2.EVENT_MOUSEMOVE:
      self._mouseMove(x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
      self._rBtnDown()

  def _lBtnDown(self, x, y):
    self.imagePointer.nowClicked = True
    if self.imagePointer.isNullSelect(): 
      # 키보드로 선입력된 키가 없다면 
      # 마우스에서 가장 가까운 거리에 있는 포인트 선택(pixel 거리가 threshold 이상이라면 선택x) 
      self.imagePointer.setSelected(self.imagePointer.getNearIdx(x, y, thresh=5)) 

    if not self.imagePointer.isNullSelect():
      self.imagePointer.addHistory()

    if self.xy.isNextChange:
      self.xy.isNextChange = False
      self.imagePointer.changePair()
      self.imagePointer.nowClicked = False
      return
    
    self.imagePointer.setPoint(x,y)
  
  def _lBtnUp(self):
    # 마우스 업일때 클릭된거 초기화해주기
    self.imagePointer.nowClicked = False
    if not self.imagePointer.isNullSelect(): 
      self.imagePointer.setSelected(None)

  def _mouseMove(self, x, y):
    if self.imagePointer.nowClicked:
      self.imagePointer.setPoint(x,y)

  def _rBtnDown(self):
    self.imagePointer.rollback()
  

    

    