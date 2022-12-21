CPSC532S Final Project
Cindy Shi, xinpingshi2015@gmail.com
Kaiyun Guo, gky0722@cs.ubc.ca

Pre-requisites: Linux (Tested on Ubuntu 18.04) NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 16) with CUDA 11.0 and cuDNN 7.5 Python (3.7.7), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), openslide (3.4.1), pandas (1.1.3), pillow (7.0.0), PyTorch (1.6.0), scikit-learn (0.22.1), scipy (1.4.1), tensorflow (1.13.1), tensorboardx (1.9), torchvision (0.7.0), captum (0.2.0), shap (0.35.0)

To execute model, download and preprocess WSI slides first and store in "WSIvectors/slide_id_name.pt"

For Attention MIL (only WSI data)
python main2.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type amil --apply_sigfeats --fusion None

For Late Fusion Attention MIL (WSI data & raw genomic data)
python main2.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --model_type amil --apply_sigfeats

For ABC (WSI data & catagorized genomic data)
python main2.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode coattn --model_type abc --fusion concat --apply_sigfeats --apply_sig