import cv2
from pathlib import Path
SHOWNAME = "main"
PAD = 50
ZOOMRANGE= 300
VISDICT = {0:"not exist", 1:"invisible", 2:"visible"}

from src.data_reader import drawBbox
from src.img_handler import addPadding, drawImg, getSkeletons
from pathlib import Path
from src.model_helper import ModelHelper
CONFIG = "src/model/golf_mobilenetv2_256x192.py"
WEIGHT = "src/model/latest.pth"
MODELWIDTH = 192
MODELHEIGHT = 256
DEVICE = "cuda:1"
SKELETONS = getSkeletons()

def mouseEvent(event, x, y, flags, param):
  pass
  # 어떤 버튼 선택했는지, 클릭 상태 

def main():
  
  imgPath = Path("data\\input\\indor_semi_best\\images\\20201123_General_003_DIS_S_F20_SS_001_3832.jpg")
  img = cv2.imread(imgPath.as_posix())
  imgH,imgW = img.shape[:2]
  x1,y1,x2,y2 = addPadding(*drawBbox(img, imgW, imgH, imgPath), padding=30, imgW=imgW, imgH=imgH)

  modelHelper = ModelHelper(CONFIG,WEIGHT, modelWidth=MODELWIDTH, modelHeight=MODELHEIGHT, device=DEVICE)
  output = modelHelper.inferenceModel(img, (x1,y1,x2,y2))
  dst = drawImg(img, output, SKELETONS)
  cv2.imshow(SHOWNAME,dst)
  cv2.waitKey()

if __name__=="__main__":
  main()