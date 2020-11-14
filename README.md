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
  1. Run **main_RGBD_Smart_woVal_twopath_toTensorChoose_n30p160_noflip.py** with one GPU or multiple GPU to train the Network. Moreover, set your certain ArgumentParser or default one.
  
      **ArgumentParser elements**
      ```python
      -sr --s <weights decay parameter> --arch <model architecture> --depth <model depth> --subject <training/testing subject in training/testing stage> --save       <path   to save the model weighting> --toTensorform <data transform: 1 means -1~1> --epochs <training epochs>
      ```

      **Command example**
      ```python
      python main_RGBD_Smart_woVal_twopath_toTensorChoose_n30p160_noflip.py -sr --s 0.0001 --arch vgg_cbam_depth_gap --depth 15 --subject SubjectA --save     ./use_now/n30p160_noflip/depth_map/toTensor0.5/first --toTensorform 1 --epochs 60
        ```


  2. Run **confusion_matrix_for_newarch.py** for getting confusion matrix for certain Subject of Dataset. And **model_best.pth.tar** is the model weights with best testing accuracy.
  
      **ArgumentParser elements**
      ```python
      --datapath <testing data location> --arch <model architecture> --depth <model depth> --subject <testing subject> --toTensorform <data_transform : 1 means     -1~1> --normalization <confusion matrix value normalization or not> --model <testing model path>
        ```

      **Command example**
      ```python
      python confusion_matrix_for_newarch.py --datapath ./RGBD_Numpy_mid_n30p160_noflip --arch vgg_cbam_depth_gap --depth 15 --subject SubjectA --toTensorform 1 --normalization 0 --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar
      ```


  3. Run **confusion_matrix_for_newarch_PR.py** for calculating precision, recall and F-score.
  
      **ArgumentParser elements**
      ```python
      --datapath <testing data location> --arch <model architecture> --depth <model depth> --subject <testing subject> --toTensorform <data_transform : 1 means -1~1> --normalization <confusion matrix value normalization or not> --model <testing model path>
      ```

      **Command example**
      ```python
      python confusion_matrix_for_newarch_PR.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectC_96.47/model_best.pth.tar --subject SubjectC --normalization 1
      ```


  4. For obtaining **gradcam**, **feature map** and **attention mask**.
  
      **GradCAM**
       * RGB stream

          **Command example**
          ```python
          python newarch_gradcam_depth_v1_RGB_getpoint.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectE_92.75/model_best.pth.tar --subject SubjectE --save test3/DSA_v1/thirdbottleneck/RGB
          ```
      * Depth stream

          **Command example**
           ```python
            python newarch_gradcam_depth_v1_depth_getpoint.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectE_92.75/model_best.pth.tar --subject SubjectE --save test4/DSA_v1/thirdbottleneck/depth
          ```
      
      **Average feature map**
       * RGB stream

          **Command example**
          ```python
          python newarch_gradcam_depth_v1_RGB_featuremap.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_2before/RGB
          ```
      * Depth stream

          **Command example**
          ```python
          python newarch_gradcam_depth_v1_depth_featuremap.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_2before/depth
         ```
     
     **Depth-attention-mask**
      * Depth stream

          **Command example**
          ```python
          python newarch_gradcam_depth_v1_depth_mask.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_3/mask
         ```

  

  
