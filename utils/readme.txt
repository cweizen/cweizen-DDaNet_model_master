## DDA_Net training
-sr --s <weights decay parameter> --arch <model architecture> --depth <model depth> --subject <training/testing subject in training/testing stage> --save <path to save the model weighting> --toTensorform <data transform: 1 means -1~1> --epochs <training epochs>

python main_RGBD_Smart_woVal_twopath_toTensorChoose_n30p160_noflip.py -sr --s 0.0001 --arch vgg_cbam_depth_gap --depth 15 --subject SubjectA --save ./use_now/n30p160_noflip/depth_map/toTensor0.5/first --toTensorform 1 --epochs 60

## get confusion matrix 
--datapath <testing data location> --arch <model architecture> --depth <model depth> --subject <testing subject> --toTensorform <data_transform : 1 means -1~1> --normalization <confusion matrix value normalization or not> --model <testing model path>

python confusion_matrix_for_newarch.py --datapath ./RGBD_Numpy_mid_n30p160_noflip --arch vgg_cbam_depth_gap --depth 15 --subject SubjectA --toTensorform 1 --normalization 0 --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar

## calculate precision, recall and f-score
--datapath <testing data location> --arch <model architecture> --depth <model depth> --subject <testing subject> --toTensorform <data_transform : 1 means -1~1> --normalization <confusion matrix value normalization or not> --model <testing model path>

python confusion_matrix_for_newarch_PR.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectC_96.47/model_best.pth.tar --subject SubjectC --normalization 1

#### about gradcam, feature map and attention mask
--depth <model_depth> --datapath <datapath> --toTensorform <data_form where 1 means from -1~1> --model <modelpath> --subject <test_subject>  --save <gradcam_saving_path>

## get gradcam of RGB stream
python newarch_gradcam_depth_v1_RGB_getpoint.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectE_92.75/model_best.pth.tar --subject SubjectE --save test3/DSA_v1/thirdbottleneck/RGB

## get gradcam of depth stream

python newarch_gradcam_depth_v1_depth_getpoint.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectE_92.75/model_best.pth.tar --subject SubjectE --save test4/DSA_v1/thirdbottleneck/depth

## get depth-attention-mask

python newarch_gradcam_depth_v1_depth_mask.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_3/mask

## get average feature map of RGB stream

python newarch_gradcam_depth_v1_RGB_featuremap.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_2before/RGB

## get average feature map of depth stream

python newarch_gradcam_depth_v1_depth_featuremap.py --toTensorform 1 --depth 15 --arch vgg_cbam_depth_gap --datapath ./RGBD_Numpy_mid_n30p160_noflip --model ./use_now/n30p160_noflip/depth_map/toTensor0.5/first/SubjectA_93.03/model_best.pth.tar --subject SubjectA --save test_featuremap/DSA_v1/bottleneck_2before/depth


