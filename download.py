from huggingface_hub import hf_hub_download

# Download the model weights file
path = hf_hub_download(
    repo_id="edadaltocg/resnet18_cifar10",
    filename="pytorch_model.bin",
    local_dir="./resnet18_cifar10_v2"   # save to this folder
)
print(f"Saved to: {path}")