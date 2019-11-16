import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


class Segmentator(object):
    def __init__(self, use_cuda=True):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.CenterCrop(size=480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, image):
        self.model.eval()

        x = self.transform(image).unsqueeze(dim=0).to(self.device)
        out = self.model(x)['out']
        return out.squeeze(dim=0).cpu().numpy()
