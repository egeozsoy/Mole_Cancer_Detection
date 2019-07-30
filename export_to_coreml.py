import torch
from onnx_coreml import convert


model = torch.load('models/pytorch_model.pt', map_location='cpu')
placeholder = torch.autograd.Variable(torch.FloatTensor(1,3, 224, 224))  # 1 will be the batch size in production
torch.onnx.export(model, placeholder, 'Model.proto', verbose=True)
mlmodel = convert('Model.proto')
mlmodel.save('coreml_model.mlmodel')