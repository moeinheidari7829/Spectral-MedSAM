# Deep Spectral Method Setup for Ultrasound Image Segmentation

This README provides instructions for Step one of Spectral-MedSAM.

## 📦 Clone the Repository

Begin by cloning the UnsupervisedSegmentor4Ultrasound repository:

```bash
git clone https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound.git
cd UnsupervisedSegmentor4Ultrasound
```


### Dataset Installation
First, download the datasets from the following sources:
- **BUSI Dataset**: [Kaggle - Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)
- **SegThy Dataset**: [SegThy Ultrasound Dataset](https://www.cs.cit.tum.de/camp/publications/segthy-dataset/)

### Data Preparation
The data structure files are in the `data_structure` folder for both datasets. Please use these files to prepare the datasets in the format expected by the Deep Spectral Method. 
- Ensure to add the correct pathes
  
### Configuration Files
Place the datasets folder inside the config folder of the Deep Spectral repository.

### Running the Deep Spectral Method
After the data is prepared and the configuration files are in place, the code is now set up to run the Deep Spectral Method. You can proceed by running the relevant scripts as instructed in the UnsupervisedSegmentor4Ultrasound repository.

### Prompt for Step 2
For Step 2 of the Deep Spectral Method, we use the images from `crf_multi_region` created in the results folder, which are the output of their first step, as our prompt.
