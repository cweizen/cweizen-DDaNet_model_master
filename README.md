# DDaNet_model
  Implementation of DDaNet with backbone VGG.
  
## Statement
  * you can find the DDaNet model in **try_place/models/**
  * you can train your Dataset with DDaNet by following the **Training Steps** shown below
  * The code is trained and tested on **ASL Dataset** in numpy file
  * If there is something wrong with the code, please contact me. Thanks !
  
## Requirements
  * **Python 3.5.6**
  * pytorch 0.3.1
  * tensorflow-gpu 1.10.0
  * matplotlib 3.0.3
  * cupy 5.4.0
  
  * **GPU environment**
    * CUDA Version 8.0.61
    * CUDNN 
    * #define CUDNN_MAJOR      5
    * #define CUDNN_MINOR      1
    * #define CUDNN_PATCHLEVEL 10
    * #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
    
## Download Dataset
  * The ASL Dataset we use can be downloaded in [here](https://drive.google.com/drive/folders/1NILKG6uPw0bDJ8y6ajXWfp9HhlaToLac?usp=sharing)
  * The paper which proposed the ASL Dataset can be found in  [**Spelling It Out: Real–Time ASL Fingerspelling Recognition**](https://empslocal.ex.ac.uk/people/staff/np331/publications/PugeaultBowden2011b.pdf) 
    * [1] Pugeault, N., and Bowden, R. (2011). Spelling It Out: Real-Time ASL Fingerspelling Recognition In Proceedings of the 1st IEEE Workshop on Consumer Depth Cameras for Computer Vision, jointly with ICCV'2011
    
## Training Steps
  * Run **main_RGBD_Smart_woVal_twopath_toTensorChoose_n30p160_noflip.py** with one GPU or multiple GPU to train the Network. Moreover, set your certain ArgumentParser or default one.
  
  **ArgumentParser elements**
  ```python
-sr --s <weights decay parameter> --arch <model architecture> --depth <model depth> --subject <training/testing subject in training/testing stage> --save <path to save the model weighting> --toTensorform <data transform: 1 means -1~1> --epochs <training epochs>
```
  

  
