# Dreamer: Dual-RIS-Aided Imager in Complementary Modes

Dreamer employs two Reconfigurable Intelligent Surfaces (RIS) operating in complementary reflection and transmission modes to vastly enhance field-of-view and perception in RF imaging. We detail how RF signals interact with complex scenes and introduce tailored illumination strategies that balance spatial resolution and coverage. A physical indoor prototype validates our design. On the reconstruction side, we propose a CNN-with-external-attention network that translates RF data into high-resolution human silhouette images. Our approach achieves an SSIM of 0.83. 

### <p align="center">| [🖨️ArXiv](https://arxiv.org/abs/2407.14820) | [📰Paper](https://ieeexplore.ieee.org/abstract/document/10944307)</p>

## Running Experiments
### Project Structure
```
The description of the files:

---predict.py                       # run predict.py to get reconsturcted images 
---train.py                         # run train.py to train the model
---my_dataset.py                    # get RF data and ground truth
---transforms.py                    # data standardization preprocessing

---pretrained:                      #  the floder, where contains the pretrained PSD2Image model we use in experiment. The download link for the file (.pth) is given.
  ---best_model.pth 

---result:                          #  the floder, where contains some outputs

---src:                             #  the floder, where contains PSD2Image model
  ---PSD2Image.py                    
  
---train_utils:                     #  the floder, where contains some functions calculating various loss, including mae, IoU, F_beta and SSIM.
  ---train_and_eval.py                

---Dataset/RF_image                 # Part of the dataset is given
```

### The demo script
---Please download the pre-trained weights [```best_model.pth```](https://drive.google.com/file/d/1EM02fjfULrlAi4YPLoEEtArUGzc1bVOn/view?usp=drive_link) and place them in the ```pretrained``` directory. As for the dataset, due to its large size, it will be released at a later stage.

```python predict.py``` is the direct method to some sample outputs. The resulting images are stored in the ```result/test``` directory.

## Citation
If you find this work useful in your research, please cite:
```txt
@article{wang2025dreamer,
  title={Dreamer: Dual-RIS-aided imager in complementary modes},
  author={Wang, Fuhai and Huang, Yunlong and Feng, Zhanbo and Xiong, Rujing and Li, Zhe and Wang, Chun and Mi, Tiebin and Qiu, Robert Caiming and Ling, Zenan},
  journal={IEEE Transactions on Antennas and Propagation},
  year={2025},
  publisher={IEEE}
}
```
