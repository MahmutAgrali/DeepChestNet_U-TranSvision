# DeepChestNet: Artificial intelligence approach for COVID-19 detection on computed tomography images

If you use this code, please cite with:

```
@article{augrali2023deepchestnet,
  title={DeepChestNet: Artificial intelligence approach for COVID-19 detection on computed tomography images},
  author={A{\u{g}}ral{\i}, Mahmut and Kilic, Volkan and Onan, Aytu{\u{g}} and Ko{\c{c}}, Esra Meltem and Ko{\c{c}}, Ali Murat and B{\"u}y{\"u}ktoka, Ra{\c{s}}it Eren and Acar, T{\"u}rker and Ad{\i}belli, Zehra},
  journal={International Journal of Imaging Systems and Technology},
  volume={33},
  number={3},
  pages={776--788},
  year={2023},
  publisher={Wiley Online Library}
}
```


*This directory contains the codes that are about the DeepChestNet project.

*************************FILES (for minbert-default-final-project)*********************	
	* config.py: This script contains the constant for training and testing.
	
	* data_generator.py: This script contains the code that has the data generator to fetch dataset.
	
	* train_test.py: This script contains the code that allows to train and test.
	
	* get_data_ready.py: This script prepares the datasets. It reads dicom file into npy file.
	
	* get_data_ready_with_augmentation.py: This scrip prepares the datasets. It reads dicom file into npy file by augmenting the images.
	
	* models.py: This script contains all models for this project.
	
	* remove_noise_dicom_images.py: This script that reads dicom images and remove their noise.	
	
	* train_test.py: This script includes code lines for training and testing.
	
	* utils.py: This script contains several useful functions.
