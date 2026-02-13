## Steps to use the code 

  1.  Create a directory /data first.

There is a ```config.py``` file which you can use to change the configurations.


To Train the model and save it do the following:
  - ```bash
    python main.py
    ```
This will train the resnet18 model and save the model as ```resnet18_cifar10.pth```

To Test the FP32 trained model performance on the test data, run the following command:
  - ```bash
    python test_fp32.py
    ```

To perform the quantization of the trained model, run the following command with the arguments according to your need ( here I am doing 8-Bit Quantization).

This code will run the quantization algo and will save the quantized model in the disk ( which can be later used ):
  - ```bash
    python perf_quantization.py --weight_quant_bits 8 --activation_quant_bits 8
    ```
To Test the performance of the Quantized model, do the following:
  - ```bash
    python test_quantize.py
    ```

### Important Note

If you are trying to run the model on a hardware device, than simply loading and running the quantized model using pytorch loading function will not work.

There are some changes that needed to be made before running the inference on an actual hardware. Once you load the model, you need to do some extra steps that are already mentioned in the ```test_quantize.py``` file.

The steps mainly involves doing CONV+BATCH norm together, which is implemented using the ```fold_batchnorms()``` function inside the ```quantize.py``` file, followed by ```swap_to_quant_modules()``` and ```load_int8_weights_into_model()``` which are also there in the file ```quantize.py``` and ```test_quantize.py``` file respectively.


Please Go through the ```test_quantize.py``` file to get an idea what to be done if you are planning to run the quantized model on accelerators.
