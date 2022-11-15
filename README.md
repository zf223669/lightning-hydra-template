# DiffMotion

The code will be coming soon!
## Environment
python=3.10.5 pytorch=1.13 pytorch-lighting=1.6.5 hydra-core=1.2.0 CUDAtoolkit=11.7
hardware Geforce RTX 3090

### Clone and download the code
'git clone https://github.com/zf223669/DiffMotion.git' 
### Setting conda environment
`conda create -n DiffusionGestureGeneration python=3.10.5`  
1: open the project in PyCharm IDE  
2: Setting the project env to DiffusionGestureGeneration

## Data Prepare
1: We used the [Trinity Speech-Gesture Dataset](https://trinityspeechgesture.scss.tcd.ie/) to train our DiffMotion models.  
2: We follow the data preparing process by [StyleGestures](https://github.com/zf223669/StyleGestures).  
3: After preparing process done, we could get the datas in folder ./StyleGestures/data/GENEA/processed.  
4: Copy all the folder and files in the processed folder to our project folder ./data/GesturesData. (need to create manually the GesturesData folder)
