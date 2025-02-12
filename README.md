# Inverse Rendering of Near-Field mmWave MIMO Radar for Material Reconstruction

https://github.com/user-attachments/assets/3ae958b9-8b7e-4e31-bc6a-e8463d6170eb

![language](https://img.shields.io/badge/language-Python-brown)
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

Source code for our paper "Inverse Rendering of Near-Field mmWave MIMO Radar for Material Reconstruction". DOI `10.1109/JMW.2025.3535077`, links to the paper will be included after publication.

# Install

Clone repository with submodules (`git clone --recursive`), or initialize submodules afterwards (`git submodule update --init --recursive`). Note that there must be a [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) capable GPU available on the target system.

Basically, we require a Python3 environment where all packages from `requirements.txt` are installed. Example installation commands on an Ubuntu-based system:

### Install dependencies

    sudo apt install python3 python3-venv

### Setup virtual environment and install required packages

    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt

Optionally patch `slangtorch` to emit the `--use_fast_math` nvcc option, which yields up to 2-3x increased runtime performance during backpropagation.

    find -O3 -L env -name "slangtorch.py" -exec sed -i 's/extra_cuda_cflags = \[\"-std=c++17\"\]/extra_cuda_cflags = \[\"-std=c++17\", \"--use_fast_math\"\]/g' {} +

# Download MAROON dataset

See [MAROON](https://github.com/vwirth/maroon) (included as a [submodule](submodules/maroon/README.md)), or download the example [MAROON Mini Dataset](https://faubox.rrze.uni-erlangen.de/getlink/fi43P9pBvMVCGz5xJSfRRM/maroon_mini.zip) and extract to a folder of choice.

# Run code

To execute differentiable radar rendering while using defaults on any dataset in MAROON, use:

    python main.py /path/to/maroon/33_s2_hand_open/30

 To use a different dataset, simply replace the respective part of the path argument, i.e. `33_s2_hand_open/30`, accordingly. For example after extracting the [MAROON Mini Dataset](https://faubox.rrze.uni-erlangen.de/getlink/fi43P9pBvMVCGz5xJSfRRM/maroon_mini.zip) to `data/maroon_mini/`:

    python main.py data/maroon_mini/02_cardboard/30

Each optimization run can be examined via [Tensorboard](https://www.tensorflow.org/tensorboard), or by looking at the respective output in `runs/`, where a folder is created for each run using the following scheme: `runs/<datetime>_<hostname>-<dataset>-<hash>`.

Exemplary results for `02_cardboard/30` in layout (depth, normals, prediction, target, error map) using default parameters:

https://github.com/user-attachments/assets/69281d20-77d9-4072-9a68-a53cc0be7970

Run using defaults, but with different loss functions, as in Figure 9:

    python main.py /path/to/maroon/33_s2_hand_open/30 --loss [l1, l1_complex, l2, l2_reco]

Run using defaults, but with different material regularization (storage) options, as in Figure 10:

    python main.py /path/to/maroon/33_s2_hand_open/30 --material_storage [global, voxelgrid, hashgrid, vertex]

Run using defaults, but with different material models, as in Figure 11:

    python main.py /path/to/maroon/33_s2_hand_open/30 --material [0-4]

Run using defaults, but with different features turned on or off, as in Figure 12:

    python main.py /path/to/maroon/33_s2_hand_open/30 [--no_emptyfiltered, --no_reg_offset, --use_normalmap]

See `main.py` for a list of all possible command line arguments.
Note that the `--use_apc` option will raise an exception per default, since this requires additional information regarding the antenna radiation pattern, which is not publicly available.

# Acknowledgments

The authors would like to thank the Rohde & Schwarz GmbH & Co. KG (Munich, Germany) for providing the radar imaging device and technical support which made this work possible.
This work was (partly) funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – SFB 1483 – Project-ID 442419336, EmpkinS.
The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR project b201dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the DFG – Project-ID 440719683.

# Citation

    TBD, will be updated after publication
