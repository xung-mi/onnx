import torch
import torch.nn as nn
import torch.nn.functional as F

class RedundantModel(nn.Module):
    def __init__(self):
        super(RedundantModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        
        # Identity (redundant node)
        x = torch.clone(x)
        
        # Transpose (redundant reshape)
        x = x.permute(0, 1, 3, 2)
        x = x.permute(0, 1, 3, 2)
        
        x = self.conv2(x)
        
        # Cast (simulated redundant op)
        x = x.float()
        
        # Reshape (redundant)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Export to ONNX
model = RedundantModel()
model.eval()
dummy_input = torch.randn(1, 3, 8, 8)

torch.onnx.export(
    model,
    dummy_input,
    "redundant_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
