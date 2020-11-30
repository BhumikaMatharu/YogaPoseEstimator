# CS256 Group B - Yoga Pose Estimation
We implemented and compared VIBE and Detectron2 (DensePose) for 3D pose estimation of yoga poses.  

We also integrated one of the models into a simple camera application that captures the live frames and predicts the yoga poses.

## Getting Started
For the camera application, detectron2 is required  
- Follow the instructions from https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md to install detectron2  
    - For Windows, follow the instruction from https://medium.com/@dgmaxime/how-to-easily-install-detectron2-on-windows-10-39186139101c  
- Run pip install to install other required packages  
- Run all cells from ```DensePose_experiment_with_Transfer_Learning_on_Simple_Dataset.ipynb``` to create and save the detectron2 model.  
    - Download the Yoga dataset being used from (Link here) and save it to the correct directory.  
- Run ```camera.py``` to start the camera application  

For VIBE yoga pose estimation, download the required packages and run  
```python identifyYogaPose.py --vid_file ./path/to/videofile```

## About
Group B submission for Final project for CS256: Selected Topic in Artificial Intelligence. Led by Instructor: Mashhour Solh, Ph.D. at San Jose State University  

Members: Duy Ngo, Mariia Surmenok, Bhumika Kaur Matharu, Kalpnil Anjan, Sushant Mane

The code maybe used for educational and commercial use under no warranties.

## Credits
AWS Educate Team - Free credits for services.



