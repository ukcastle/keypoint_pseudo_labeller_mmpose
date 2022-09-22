import cv2

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

def drawFullCrossLine(img, x, y, imgW, imgH, color):
  cv2.line(img, (x,0), (x, imgH), color)
  cv2.line(img, (0,y), (imgW, y), color)
  return img

def addPadding(x1, y1, x2, y2, padding, imgW, imgH):
  x1, y1 = (max(p-padding, 0) for p in (x1, y1))
  x2, y2 = (min(p+padding, length) for p, length in ((x2,imgW), (y2, imgH)))
  return x1, y1, x2 ,y2