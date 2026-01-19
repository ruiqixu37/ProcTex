# ProcTex
Official implementation of **ProcTex: Consistent and Interactive Text-to-texture Synthesis for Part-based Procedural Models** (Eurographics 2026)

## Environment Setup

The implementation is tested on the following configurations:

* Ubuntu 22.04 with NVIDIA RTX 4090 (CUDA 11.8, Python 3.10, PyTorch 2.1.0)
* Ubuntu 24.04 with NVIDIA RTX 5090 (CUDA 12.8, Python 3.11, PyTorch 2.8.0)

Other configurations may work but have not been tested.

### Installation

1. **Create a conda environment and install PyTorch:**

```bash
# CUDA 11.8
conda create -n proctex python=3.10
conda activate proctex
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.8
conda create -n proctex python=3.11
conda activate proctex
pip install torch==2.8.0 torchvision
```

2. **Install PyTorch3D:**
```bash
# Option 1: Install from pre-built wheels (recommended)
# CUDA 11.8
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# CUDA 12.8
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.8.0cu128

# Option 2: Build from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

4. **Install nvdiffrast:**
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

5. **Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

## Data Preparation

### Training Data Structure
Organize your training data as follows:
```
data/
└── procedural_<object>/
    └── train/
        ├── 0.obj          # Mesh file
        ├── 0.json         # Procedural parameters
        ├── 1.obj
        ├── 1.json
        └── ...
```

Each `.json` file should contain the procedural parameters used to generate the corresponding `.obj` mesh.

### Inference Data Structure
```
data/
└── procedural_<object>/
    └── inference/
        ├── 0.obj
        ├── 0.json
        └── ...
```

## Training

### Configuration

Create or modify a config file in `configs/`. See `configs/sofa.yaml` as an example.


### Run Training

```bash
python train.py --config configs/sofa.yaml
```

Training outputs will be saved to the specified `outdir` (e.g., `logs/sofa/`), including:
- `out_texture_group*.png`: Generated texture maps for each component group
- `ckpt/`: Trained model checkpoints
- `component_correspondence.json`: Component correspondence mapping
- `mesh*_component*.obj`: Mesh components with applied textures for each training mesh

## Inference

After training, you can apply the learned texture to new procedural mesh variations.

### Configure Inference

Edit the paths in `inference.py`:

```python
TARGET_MESH_FP = "data/procedural_sofa/train/41.obj"  # Template mesh used in training
TRAIN_RESULT_FP = "logs/sofa/"                        # Training output directory
INFERENCE_DATA_DIR = "data/procedural_sofa/inference" # New meshes for inference
```

### Run Inference

```bash
python inference.py
```

Results will be saved to `<TRAIN_RESULT_FP>/hue/`, including:
- `mesh*.obj`: Textured mesh files
- `mesh*_group*.png`: Transferred texture maps
- `component_correspondence.json`: Component correspondence for each mesh

## Visualization

You can generate a video to visualize the inference results across different mesh variations.

### Configure Video Generation

Edit the paths and camera settings in `produce_video.py`:

```python
PARAM_DIR = "data/procedural_sofa/inference"  # Directory containing mesh and parameter files
RESULT_DIR = "logs/sofa/"                      # Training/inference output directory
OUTPUT_PATH = "sofa.mp4"                       # Output video path
TURNTABLE_VIDEO = True                         # True: rotating view, False: static view
```

You may also need to adjust the camera settings depending on your mesh size and orientation:

```python
# Camera settings
radius = 1      # Distance from camera to object (increase if mesh appears too large)
fovy = 50       # Field of view in degrees
```

The camera elevation is set in the `orbit_camera` call. To change it, modify:

```python
pose = orbit_camera(-30, rot_angle, radius=radius, is_degree=True)
#                   ^^^
#                   Elevation angle in degrees (negative = looking down)
```

### Generate Video

```bash
python produce_video.py
```

This will produce an MP4 video showing each mesh variation with its procedural parameters overlaid. When `TURNTABLE_VIDEO=True`, the camera rotates progressively across different meshes, creating a turntable effect.

## Citation

If you find this work useful, please cite:
```bibtex
@article{xu2025proctex,
  title={ProcTex: Consistent and Interactive Text-to-texture Synthesis for Part-based Procedural Models},
  author={Xu, Ruiqi and Zhu, Zihan and Ahlbrand, Benjamin and Sridhar, Srinath and Ritchie, Daniel},
  journal={arXiv preprint arXiv:2501.17895},
  year={2025}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
