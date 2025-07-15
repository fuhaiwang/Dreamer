# Dreamer
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
---Please download the pre-trained weights [best_model.pth](https://drive.google.com/file/d/1EM02fjfULrlAi4YPLoEEtArUGzc1bVOn/view?usp=drive_link) and place them in the pretrained directory. As for the dataset, due to its large size, it will be released at a later stage.
```python predict.py``` is the direct method to some sample output.

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
