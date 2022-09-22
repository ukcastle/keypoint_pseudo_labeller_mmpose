import cv2
import torch
from mmpose.apis import init_pose_model
from mmpose.core.bbox import bbox_xywh2cs
import torchvision
KEYPOINTS = [
          "left_shoulder",
          "right_shoulder",
          "left_elbow",
          "right_elbow",
          "left_wrist",
          "right_wrist",
          "left_hip",
          "right_hip",
          "left_knee",
          "right_knee",
          "left_ankle",
          "right_ankle",
          "grip",
          "shaft_end",
          "club_head" #총 15개
        ]

from src.util import timeit

class ModelHelper:
  @timeit
  def __init__(self, configPath, weightPath, modelWidth=192, modelHeight=256, device="cuda:1"):
    self.model = init_pose_model(configPath, weightPath, device)
    self.model.eval()
    self.transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.model.forward = ModelHelper.forward_dummy
    self.device = device
    self.modelWidth = modelWidth
    self.modelHeight = modelHeight
    self.center, self.scale = (bbox_xywh2cs((0,0,self.modelWidth,self.modelHeight), self.modelWidth/self.modelHeight))
  
  @staticmethod
  def forward_dummy(model, img, meta):
    output = model.backbone(img)
    if model.with_neck:
        output = model.neck(output)
    
    output_heatmap = model.keypoint_head.inference_model(output, flip_pairs=None)

    keypoint_result = model.keypoint_head.decode(
        meta, output_heatmap, img_size=meta[0]["size"])

    return torch.tensor(keypoint_result["preds"])

  # @timeit
  def resizeWithLetterBox(self, inputImg):
    img = inputImg.copy()
    cropH, cropW = img.shape[:2]
    ratio = min(self.modelWidth/cropW, self.modelHeight/cropH)
    refineW, refineH = (int(x*ratio) for x in (cropW, cropH))
    dw = (self.modelWidth - refineW)/2
    dh = (self.modelHeight - refineH)/2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, (refineW, refineH), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, left, top, ratio

  # @timeit
  def makeInput(self, img):
    outTensor = self.transform(img)
    outTensor = outTensor.unsqueeze(0)
    outTensor = outTensor.to(self.device)
    meta = [{
      "image_file" : "image",
      "center" : self.center,
      "scale" : self.scale,
      "size" : (self.modelWidth, self.modelHeight)
    }]

    return outTensor, meta
  
  @timeit
  def inference(self, inputTensor, meta):
    output = ModelHelper.forward_dummy(self.model, inputTensor, meta)
    return output
  
  # @timeit
  def refineOutput(self, output, leftPad, topPad, resizeRatio, x1, y1):
    output[:,:,0] = ((output[:,:,0] - leftPad) / resizeRatio)
    output[:,:,1] = ((output[:,:,1] - topPad) / resizeRatio)
    output[:,:,0] += x1
    output[:,:,1] += y1

    return output[0].to(torch.int32).numpy().tolist()

  def inferenceModel(self, img, bbox):

    x1,y1,x2,y2 = bbox
    cropImg = img[y1:y2, x1:x2]
    cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
    cropImg, leftPad, topPad, resizeRatio = self.resizeWithLetterBox(cropImg)
    inputTensor, meta = self.makeInput(cropImg)
    output = self.inference(inputTensor, meta)
    refinedOutput = self.refineOutput(output, leftPad, topPad, resizeRatio, x1, y1)
    
    return refinedOutput