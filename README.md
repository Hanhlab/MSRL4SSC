# MSRL4SSC: A Multi-Source Satellite Image Representation Learning Framework for Soil Salt Content Retrieval

This repository provides the implementation of **MSRL4SSC**, a multi-source satellite representation learning framework for soil salt content (SSC) retrieval. The framework integrates cross-modal self-supervised pretraining, modality-optional task adaptation, and regression-based SSC retrieval using Sentinel-1 (S1) and Sentinel-2 (S2) imagery.

## Requirements

```bash
conda activate torch_suc
pip install -r requirements.txt
```

## Training

### Stage 1: Cross-Modal Self-Supervised Pretraining

### Stage 2: Modality-Optional Task Adaptation

Run both stages using:

```bash
python train.py \
  --data-root /path/to/dataset \
  --pretrained-checkpoint /path/to/CROMA_base.pt \
  --base-config configs/five_index_full.yaml \
  --output-root outputs
```

## Stage 3: SSC Retrieval

The learned point representations are reduced using principal component analysis (PCA) and combined with conventional predictors for SSC retrieval using ten regression models.

```bash
python evaluate_stage3.py \
  --feature-root outputs/selected_features \
  --sample-index outputs/master_index_seed74.csv \
  --output-dir outputs/stage3
```

## Acknowledgements

We are deeply grateful for the work of these researchers.
@inproceedings{Fuller2023CROMA,
	author = {Fuller, Anthony and Millard, Koreen and Green, James},
	booktitle = {Advances in Neural Information Processing Systems},
	doi = {10.52202/075280-0241},
	editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
	pages = {5506--5538},
	publisher = {Curran Associates, Inc},
	title = {{CROMA}: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders},
	url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf},
	volume = {36},
	year = {2023}
}

@inproceedings{Wang2023multi,
	title={Multi-modal learning with missing modality via shared-specific feature modelling},
	author={Wang, Hu and Chen, Yuanhong and Ma, Congbo and Avery, Jodie and Hull, Louise and Carneiro, Gustavo},
	booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages={15878--15887},
	year={2023},
	organization={IEEE},
	doi = {10.1109/CVPR52729.2023.01524}
}

## Acknowledgements

We sincerely thank the authors of the following works for their valuable contributions and publicly available research resources, which have inspired and supported the development of MSRL4SSC.

- @inproceedings{Fuller2023CROMA,
	author = {Fuller, Anthony and Millard, Koreen and Green, James},
	booktitle = {Advances in Neural Information Processing Systems},
	doi = {10.52202/075280-0241},
	editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
	pages = {5506--5538},
	publisher = {Curran Associates, Inc},
	title = {{CROMA}: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders},
	url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf},
	volume = {36},
	year = {2023}
}
-@inproceedings{Wang2023multi,
	title={Multi-modal learning with missing modality via shared-specific feature modelling},
	author={Wang, Hu and Chen, Yuanhong and Ma, Congbo and Avery, Jodie and Hull, Louise and Carneiro, Gustavo},
	booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages={15878--15887},
	year={2023},
	organization={IEEE},
	doi = {10.1109/CVPR52729.2023.01524}
}
