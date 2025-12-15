import torch as th
import torchvision.models as models
import torch.nn as nn

# 채널 평균 모듈 정의
class ChannelAveraging(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)

# ResNet18 불러오기
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet_layers = nn.Sequential(*list(resnet.children())[:-2])  # 마지막 avgpool, fc 제거

channel_avg = ChannelAveraging()
flatten = nn.Flatten()

# 더미 입력 (배치 1, 채널 3, 480x640)
dummy_input = th.zeros(1, 3, 480, 640)

with th.no_grad():
    x = dummy_input
    print("입력:", x.shape)

    x = resnet_layers(x)
    print("ResNet 출력:", x.shape)  # [1, 512, 15, 20]

    x = channel_avg(x)
    print("채널 평균 후:", x.shape)  # [1, 1, 15, 20]

    x = flatten(x)
    print("Flatten 후:", x.shape)  # [1, 300]
