import argparse
import torch
from copy import deepcopy
from models.experimental import attempt_load
from models.yolo import BaseModel, parse_model, Detect, DDetect, Segment, Panoptic, DualDetect, TripleDetect, \
    DualDDetect, TripleDDetect
from pathlib import Path
from utils.downloads import attempt_download
from utils.general import LOGGER, check_yaml, print_args, intersect_dicts
from utils.torch_utils import initialize_weights, select_device, scale_img
from models.common import Concat, RepNCSPELAN4
import torch.nn.functional as F

class DetectionModelBackboneFeature(BaseModel):
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect, Segment, Panoptic)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Panoptic)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once
        if isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # forward = lambda x: self.forward(x)[0][0] if isinstance(m, (DualSegment, DualPanoptic)) else self.forward(x)[0]
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        # initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        x = self.model[0](x)  # Silence layer
        x = self.model[1](x)  # First Conv layer
        x = self.model[2](x)  # Second Conv layer
        x = self.model[3](x)  # First RepNCSPELAN4 block
        x = self.model[4](x)  # ADown layer
        x = self.model[5](x)
        x = self.model[6](x)
        x = self.model[7](x)
        x = self.model[8](x)
        x = self.model[9](x)  # Last layer of the backbone
        return x


class DetectionModelCombined(BaseModel):
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None, backbone_features=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        self.backbone_features = backbone_features  # Store the provided backbone features

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect, Segment, Panoptic)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Panoptic)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        if isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        self.model = self.model.eval()
        self.info()
        LOGGER.info('')

    def forward(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        mock_input_shapes = [torch.Size([1, 3, 256, 256]), torch.Size([1, 3, 32, 32])]
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if isinstance(x, torch.Tensor):
                x_in = x.shape
                if self.backbone_features is not None and x_in == self.backbone_features.shape and x_in not in mock_input_shapes:
                    x = self.backbone_features
                    self.backbone_features = None
                    # print(f"Replacing: Input shape matches backbone_features.shape")
            y.append(x if m.i in self.save else None)  # save output

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='detect/yolov9-c.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)
    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = DetectionModelBackboneFeature(opt.cfg).to(device)
    weights = "../weights/best.pt"
    ckpt = torch.load(attempt_download(weights), map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
    model.load_state_dict(csd, strict=False)
    model.eval()
    y = model(im)
    print(f'Done. {y.shape}')
