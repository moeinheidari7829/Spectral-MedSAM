# Step2: BUSI and Thyroid Experiments

This folder contains code for BUSI and Thyroid segmentation experiments for the BMEG591T (ML in Medicine) course project (2024W2).

## Getting Started

### 1. Clone the repository
```
git clone https://github.com/moeinheidari7829/Spectral-MedSAM.git
cd Spectral-MedSAM/Step2
```

### 2. Install dependencies
It is recommended to use Anaconda:
```
conda create -n samed python=3.8
conda activate samed
pip install -r requirements.txt
```

### 3. Download SAM Checkpoint
Download the SAM checkpoint from [this link](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view) and place it in the `checkpoints/` folder:
```
mkdir -p checkpoints
# Download and move the checkpoint file here
```

---

## Running BUSI Experiments
1. Prepare your BUSI dataset and update the paths in `run_busi_experiment.sh` as needed.
2. Run the experiment:
```
bash run_busi_experiment.sh
```
3. Results and TensorBoard logs will be saved in the specified output directory.

## Running Thyroid Experiments
1. Prepare your Thyroid dataset and update the paths in `run_thyroid_experiments.sh` as needed.
2. Run the experiment:
```
bash run_thyroid_experiments.sh
```
3. Results and TensorBoard logs will be saved in the specified output directory.

## Visualizing Results
To visualize training progress and predictions:
```
tensorboard --logdir <output_dir> --port 6006
```
Replace `<output_dir>` with your experiment's output folder, e.g. `final_output_thyroid_new_vis/thyroid` or `final_output_busi/`.

---

## Notes
- Make sure the checkpoint file is named as expected by your scripts (e.g., `sam_vit_b_01ec64.pth`) and placed in the `checkpoints/` directory.
- You may need to adjust dataset paths and experiment parameters in the shell scripts for your environment.

## Citation & License
See the main repo for citation and license information.

---

For any questions, please open an issue in the GitHub repo.



## Acknowledgments

We gratefully acknowledge the contributions of the following projects, which played a key role in enabling this work:

- **[SAMed](https://github.com/hitachinsk/SAMed)**: We utilized the SAMed model as the main part of our methodology for fine-tunning.

- **[UnsupervisedSegmentor4Ultrasound](https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound)**: We used this repository to generate the clusters in the first step.

Special thanks to the developers and contributors of both repositories for their impactful work and open-source spirit.
