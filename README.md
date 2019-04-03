NConv-CNN on NYU-Depth-v2
============================

The repo provides an implementation to train/test our method ["Confidence Propagation through CNNs for Guided Sparse Depth Regression"](https://arxiv.org/abs/1811.01791) on the ["NYU-Depth-v2 dataset"](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

This repo is forked from the PyTorch implementation for ["Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"](https://arxiv.org/pdf/1709.07492.pdf) by [Fangchang Ma](http://www.mit.edu/~fcma) and [Sertac Karaman](http://karaman.mit.edu/).

We provide training for both networks `Enc-Dec-Net[EF]` and `MS-Net[LF]` on the RGB-D input of the dataset.
as they were described in the paper.

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Testing](#testing)
0. [Citation](#citation)

## Requirements
This code was tested with Python 3 and PyTorch 1.0.
- Install [PyTorch](http://pytorch.org/) on a machine with CUDA GPU.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and other dependencies (files in our pre-processed datasets are in HDF5 formats).
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py matplotlib imageio scikit-image opencv-python
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 formats, and specify the path to the datasets in `create_data_loaders()` in `main.py` The downloading process might take an hour or so. The NYU dataset requires 32G of storage space.
	```bash
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz	
	```
## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python main.py --help
```

For instance, run the following command to train the network `Enc-Dec-Net[EF]`, and both RGB and 100 random sparse depth samples as the input to the network.
```bash
python main.py -a guided_enc_dec -m rgbd -s 100 --data nyudepthv2 --optimizer adam --lr 0.001 --lr-decay 10
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python main.py --evaluate [path_to_trained_model]
```

## Citation
If you use the code or method in your work, please consider citing the original authors of the code:

	@article{Ma2017SparseToDense,
		title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
		author={Ma, Fangchang and Karaman, Sertac},
		booktitle={ICRA},
		year={2018}
	}
	@article{ma2018self,
		title={Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera},
		author={Ma, Fangchang and Cavalheiro, Guilherme Venturelli and Karaman, Sertac},
		journal={arXiv preprint arXiv:1807.00275},
		year={2018}
	}

And our paper:
```
@article{eldesokey2018confidence,
  title={Confidence Propagation through CNNs for Guided Sparse Depth Regression},
  author={Eldesokey, Abdelrahman and Felsberg, Michael and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:1811.01791},
  year={2018}
}
```
