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

class ModelHelper:
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
  
  @staticmethod
  def forward_dummy(model, img, meta):
    output = model.backbone(img)
    if model.with_neck:
        output = model.neck(output)
    
    output_heatmap = model.keypoint_head.inference_model(output, flip_pairs=None)

    keypoint_result = model.keypoint_head.decode(
        meta, output_heatmap, img_size=meta[0]["size"])

    return torch.tensor(keypoint_result["preds"])

  def inferenceModel(self, img, bbox):
    x1,y1,x2,y2 = bbox
    cropImg = img[y1:y2, x1:x2].copy()
    cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
    cropH, cropW = cropImg.shape[:2]
    ratio = min(self.modelWidth/cropW, self.modelHeight/cropH)
    refineW, refineH = (int(x*ratio) for x in (cropW, cropH))
    dw = (self.modelWidth - refineW)/2
    dh = (self.modelHeight - refineH)/2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    cropImg = cv2.resize(cropImg, (refineW, refineH), interpolation=cv2.INTER_LINEAR)
    cropImg = cv2.copyMakeBorder(cropImg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    cropImg = self.transform(cropImg)
    cropImg = cropImg.unsqueeze(0)
    cropImg = cropImg.to(self.device)
    center, scale = (bbox_xywh2cs((0,0,self.modelWidth,self.modelHeight), self.modelWidth/self.modelHeight))

    meta = [{
      "image_file" : "image",
      "center" : center,
      "scale" : scale,
      "size" : (self.modelWidth, self.modelHeight),
      "flip_pairs" : []
    }]

    output = ModelHelper.forward_dummy(self.model, cropImg, meta)
    output[:,:,0] = ((output[:,:,0] - left) / ratio).to(torch.int16)
    output[:,:,1] = ((output[:,:,1] - top) / ratio).to(torch.int16)
    output[:,:,0] += x1
    output[:,:,1] += y1
    
    return output