# AR-CNN, Fast AR-CNN

This repository is implementation of the "Deep Convolution Networks for Compression Artifacts Reduction". <br />
In contrast with original paper, It use RGB channels instead of luminance channel in YCbCr space and smaller(16) batch size.

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

<table>
    <tr>
        <td><center>Input</center></td>
        <td><center>JPEG (Quality 10)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_jpeg_q10.png" height="300"></center>
    	</td>
    </tr>
    <tr>
        <td><center>AR-CNN</center></td>
        <td><center>Fast AR-CNN</center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_ARCNN.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_FastARCNN.png" height="300"></center>
        </td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
Data augmentation option **--use_augmentation** performs random rescale and rotation. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python main.py --arch "ARCNN" \     # ARCNN, FastARCNN
               --images_dir "" \
               --outputs_dir "" \
               --jpeg_quality 10 \
               --patch_size 24 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 5e-4 \
               --threads 8 \
               --seed 123 \
               --use_augmentation \
               --use_fast_loader              
```

### Test

Output results consist of image compressed with JPEG and image with artifacts reduced.

```bash
python example --arch "ARCNN" \     # ARCNN, FastARCNN
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \
               --jpeg_quality 10               
```
