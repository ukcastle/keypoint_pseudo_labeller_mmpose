from pathlib import Path
from .model_helper import KEYPOINTS
from .img_handler import getSkeletons

DEFAULTDATE = "2022-09-20 00:00:00"

class COCO_dict(dict):
  def __init__(self,*args, **kwargs):
    super().__init__(*args, **kwargs)
    self["images"] = []
    self["annotations"] = []
    self.updateInfo()
    self.updateCategories()
    self.updateLicense()

    self.bboxList = []

  def updateInfo(self):
    self.update({
      "info":{
        "year":2022,
        "version":"1.0",
        "description" : "golf keypoint",
        "contributor" : "josw631@gmail.com",
        "url" : ".",
        "date_created" : DEFAULTDATE
      }
    })

  @staticmethod
  def refineSkeletons(): # [[startIdx, endIdx, color], [...]] -> [[startIdx, endIdx], [...]]
    return [[s,e] for s,e,_ in getSkeletons()]

  def updateCategories(self):
    self.update({
      "categories" : [{
        "supercategory": "person",
        "id": 1,
        "name": "golfer",
        "keypoints":KEYPOINTS,
        "skeleton": COCO_dict.refineSkeletons()
      }]
    })

  def updateLicense(self):
    self.update({
      "licenses": [
        {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},
        {"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},
        {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},
        {"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},
        {"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},
        {"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},
        {"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}
      ]
    })

  @staticmethod
  def outputList2keyArray(outputList): # [[x,y,vis],[x,y,vis]...] -> [x,y,vis,x,y,vis]
    keyArray = []

    for x,y,vis in outputList:
      keyArray.append(x)
      keyArray.append(y)
      keyArray.append(vis)

    return keyArray

  def updateDict(self, imgPath : Path, outputList, bbox, imgW, imgH):
    curImageDict = self.makeImageDict(imgPath.as_posix(), imgW, imgH)
    self["images"].append(curImageDict)
    self["annotations"].append(self.makeAnnsDict(COCO_dict.outputList2keyArray(outputList), curImageDict["id"], *bbox))

    self.bboxList.append({
      "bbox" : list(bbox),
      "category_id" : 1,
      "image_id" : curImageDict["id"],
      "score" : 1.0
    })
    
  def makeAnnsDict(self, keyArr, imageId, x1, y1, w, h):
    return {
      "id": len(self["annotations"]), 
      "image_id": imageId, 
      "category_id": 1, 
      "segmentation": [], 
      "area": float(w*h), 
      "bbox": [x1,y1,w,h], 
      "iscrowd": 0,
      "keypoints": keyArr, 
      "num_keypoints": len(self["categories"][0]["keypoints"]),
    }
  
  def makeImageDict(self, imgPath, imgW, imgH):
    return {
      "id":len(self["images"]), 
      "width": imgW, 
      "height": imgH, 
      "file_name": imgPath, 
      "license": 1, 
      "flickr_url": "", 
      "coco_url": "", 
      "date_captured": DEFAULTDATE
    }