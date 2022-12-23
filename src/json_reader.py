from pathlib import Path
from .model_helper import KEYPOINTS
import json
from .img_handler import getKeyDict
IDX_MATCH = {
  "head"          : 0,
  "neck"          : 1,
  "right_shoulder": 3,
  "left_shoulder" : 4,
  "right_elbow"   : 5,
  "left_elbow"    : 6,
  "right_wrist"   : 7,
  "left_wrist"    : 8,
  "right_hip"     : 10,
  "left_hip"      : 11,
  "right_knee"    : 12,
  "left_knee"     : 13,
  "right_ankle"   : 14,
  "left_ankle"    : 15,
}

def getJsonPathByImgPath(imgPath):
  pathList = Path(imgPath).parts
  changeIdx, removeIdx = [pathList.index(x) for x in ["input", "images"]]
  jsonPath = Path(
    "/".join([
      *pathList[:changeIdx],"json", 
      *pathList[changeIdx+1:removeIdx], 
      *pathList[removeIdx+1:]
    ])
  ).with_suffix(".json")

  return jsonPath
def getLabelFromJson(jsonPath : Path):
  keyPointDict = {x:None for x in KEYPOINTS}
  try:
    with jsonPath.open("r", encoding="UTF8") as f:
      curJson = json.load(f)
    for curAnno in curJson["annotations"]:
      curAnno : dict
      if "points" not in curAnno.keys():
        continue
      for key, idx in IDX_MATCH.items():
        x,y,vis = curAnno["points"][idx*3:idx*3+3]
        keyPointDict[key] = [int(x/2), int(y/2), vis] # 이미지 해상도 절반으로 줄임
  except Exception as e:
    print(e)
  finally:
    return keyPointDict

from .point import ImagePointer
KEYDICT = getKeyDict()
def parseOutputList(points : ImagePointer, keyPointDict : dict):
  for key, val in keyPointDict.items():
    if val is None:
      continue
    x, y, vis = val
    curIdx = KEYDICT[key]
    points._addHistory(curIdx)
    points.setPoint(x, y, vis=vis, seletedIdx=curIdx)
  
    
    