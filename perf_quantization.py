
import torch
from config import *
from data import get_dataloaders
from model import resnet18_cifar

import torch
from quantize import *
import argparse

@torch.no_grad()
def load_int8_weights_into_model(model, int8_payload):
    for name, m in model.named_modules():
        if name in int8_payload:
            data = int8_payload[name]

            w_int = data["weight_int"].float()
            scale = data["scale"]
            zp = data["zp"]

            # dequantize ONCE
            w_fp = (w_int - zp) * scale
            m.weight.copy_(w_fp)

            if m.bias is not None and data["bias"] is not None:
                m.bias.copy_(data["bias"])


@torch.no_grad()
def test(model,testloader,device):
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantization')
    parser.add_argument('--weight_quant_bits',type=int,default=8,help='Bits to Quantize the weights')
    parser.add_argument('--activation_quant_bits',type=int,default=8,help='Activation quantization bits')
    parser.add_argument('--save_path',type=str,default='./resnet18_cifar10_quantized.pth')

    args = parser.parse_args()

    trainloader, testloader = get_dataloaders()
    model = resnet18_cifar().to(device)
    model.load_state_dict(torch.load(ckpt_path,weights_only=True))
    model.eval()

    test_acc = test(model,testloader, device)
    print(f"\nTest Acc={test_acc:.2f}%\n")

    # performing Quantization
    weight_quantize_bits = args.weight_quant_bits
    act_quantize_bits = args.activation_quant_bits

    fold_batchnorms(model)

    swap_to_quant_modules(model, weight_bits=weight_quantize_bits, act_bits=act_quantize_bits, activations_unsigned=True)
    model.to(device)
    with torch.no_grad():
        for i,(x,_) in enumerate(trainloader):
            x = x.to(device)
            y = model(x)
            if i>=100: # calibration with 5 batches
                break

    freeze_all_quant(model)
    quantize_test_acc = test(model,testloader, device)
    print(f"Quantized Test Acc={quantize_test_acc:.2f}%\n")
    print_compression(model,weight_bits=weight_quantize_bits)

    model.cpu()
    model.eval()

    int8_payload = export_int8_model(model,weight_bits=weight_quantize_bits)

    torch.save(
    {
        "int8_weights": int8_payload,
        "weight_bits": weight_quantize_bits,
    },
    args.save_path
    )

    payload = torch.load("resnet18_cifar10_quantized.pth", map_location=device,weights_only=True)

    int8_weights = payload["int8_weights"]
    weight_bits = payload["weight_bits"]
    model_int8 = resnet18_cifar()
    
    model_int8.eval()
    fold_batchnorms(model_int8)

    swap_to_quant_modules(model, weight_bits=weight_quantize_bits, act_bits=act_quantize_bits, activations_unsigned=True)

    load_int8_weights_into_model(model_int8,int8_weights)
    model_int8.to(device)
    acc = test(model_int8, testloader, device=device)
    print(f"INT8-loaded model accuracy = {acc:.2f}%")
