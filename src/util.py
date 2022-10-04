import time

def timeit(func):
  def wrapper(*args, **kwargs):
    s = time.time()
    result = func(*args, **kwargs)
    e = time.time()

    print(f"function : {func.__name__} | time : {(e-s):2.4f}")

    return result
  return wrapper

import torch
def convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
      module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                            module.momentum, module.affine,
                                            module.track_running_stats)
      if module.affine:
        module_output.weight.data = module.weight.data.clone().detach()
        module_output.bias.data = module.bias.data.clone().detach()
        # keep requires_grad unchanged
        module_output.weight.requires_grad = module.weight.requires_grad
        module_output.bias.requires_grad = module.bias.requires_grad
      module_output.running_mean = module.running_mean
      module_output.running_var = module.running_var
      module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
      module_output.add_module(name, convert_batchnorm(child))
    del module
    return module_output
    
from mmpose.core.post_processing import flip_back
class Onnx_Wrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
    
  def forward(self, img):
    output = self.model.backbone(img)
    if self.model.with_neck:
        output = self.model.neck(output)
    if self.model.with_keypoint:
        output = self._headInference(output, flip_pairs=None)
    return output

  def _headInference(self, x, flip_pairs):
    output = self.model.keypoint_head.forward(x)
    if flip_pairs is not None:
      output_heatmap = flip_back(
        output.detach().cpu(),
        flip_pairs,
        target_type=self.target_type)
      # feature is not aligned, shift flipped heatmap for higher accuracy
      if self.test_cfg.get('shift_heatmap', False):
        output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
    else:
        output_heatmap = output.detach().cpu()
    return output_heatmap