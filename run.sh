# train video model

#python main_vid.py --config ./config/ol_RGB/train_video_feature.yaml --work-dir X3D_DNLA_Mu

# test video model
python main_vid.py --config ./config/ol_RGB/test_video_feature.yaml --work-dir new_test

# train video model   - MTL-AQA dataset

#python main_vid.py --config ./config/MTL/train_video_feature.yaml --work-dir trial42


# train skel msg3d model
#python main_vid.py --config ./config/ol_RGB/train_skel_msg3d.yaml --work-dir trial75

# test skel msg3d model
#python main_vid.py --config ./config/ol_RGB/test_skel_msg3d.yaml --work-dir new_test

# train 2s model

#python main_vid.py --config ./config/ol_RGB/train_2stream.yaml --work-dir trial61

# train video model   - UNLV_dive dataset

#python main_vid.py --config ./config/MIT_skate/train_video_feature.yaml --work-dir trial66

# train skel model   - Female split

#python main_vid.py --config ./config/ol_RGB/train_skel_female.yaml --work-dir trial67


# train skel model   - Male split

#python main_vid.py --config ./config/ol_RGB/train_skel_male.yaml --work-dir trial68

#python main_vid.py --config ./config/ol_RGB/train_video_male.yaml --work-dir trial69

#python main_feat.py --config ./config/ol_feat/test_joint_feat.yaml --work-dir Test2
#python main_feat.py --config ./config/ol_feat/train_joint_feat.yaml --work-dir trial72
