dependencies = ['torch']

import torch
import torchvision

def test_model(pretrained=False):
    model = torchvision.models.convnext_tiny(pretrained=False)

    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://github.com/gholste/torchhub_test/releases/download/v1.0/weights.pt', progress=False)['weights']
        msg = model.load_state_dict(weights, strict=True)
        print(msg)

    return model