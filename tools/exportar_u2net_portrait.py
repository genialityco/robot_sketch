#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Genera models/u2net_portrait.onnx (el boceto local, ver local_sketch.py) en
una PC nueva: baja los pesos originales de U²-Net Portrait
(https://github.com/xuebinqin/U-2-Net) y los exporta a ONNX.

Necesita torch y gdown SOLO para esto (la app despues usa onnxruntime). Mejor
en un entorno aparte, para no sumarle torch al de la app:

    python -m venv exportenv
    exportenv\\Scripts\\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    exportenv\\Scripts\\python -m pip install gdown onnx
    exportenv\\Scripts\\python tools\\exportar_u2net_portrait.py

Tarda un par de minutos (baja ~176 MB).
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(HERE, 'models', '_export')
WEIGHTS_URL = 'https://drive.google.com/uc?id=1IG3HdpcRiDoWNookbncQjeaPN28t90yW'   # u2net_portrait.pth (repo oficial)
MODEL_DEF_URL = 'https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net.py'
OUT = os.path.join(HERE, 'models', 'u2net_portrait.onnx')


def main():
    import gdown
    import torch
    os.makedirs(WORK, exist_ok=True)
    weights = os.path.join(WORK, 'u2net_portrait.pth')
    if not os.path.exists(weights):
        gdown.download(WEIGHTS_URL, weights, quiet=False)
    model_def = os.path.join(WORK, 'u2net.py')
    if not os.path.exists(model_def):
        urllib.request.urlretrieve(MODEL_DEF_URL, model_def)
    sys.path.insert(0, WORK)
    from u2net import U2NET

    class Main(torch.nn.Module):
        """Solo la salida principal (d1), como la usa u2net_portrait_demo.py."""
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0]

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.eval()
    torch.onnx.export(Main(net), torch.randn(1, 3, 512, 512), OUT, input_names=['input'], output_names=['d1'],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print('listo: %s (%.0f MB)' % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
