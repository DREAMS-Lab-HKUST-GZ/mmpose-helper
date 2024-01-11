from mmpose_helper import MMPoseHelper

# helper
mm = MMPoseHelper(skeleton_style='mmpose')

# process image
img_path = "./data/test01.png"
mmpose_out = mm.process_one_image(img_path)

# visualize
import cv2
img = cv2.imread(img_path)
canvas = mm.visualize(img, mmpose_out)
cv2.imwrite(img_path.replace('.png', '.mmpose.png'), canvas)

# save to json
prediction = mmpose_out['pred_instances']
mmpose_fn = img_path.replace('.png', '.mmpose.json')
import json
with open(mmpose_fn, 'w') as fp:
    json.dump((prediction), fp)

# mmpose 133 to openpose 118 for easymocap
import numpy as np
kpts2d_mm = np.concatenate([np.array(prediction[0]['keypoints']), 
                            np.array(prediction[0]['keypoint_scores'])[:, None]], axis=-1)
kpts2d_op = MMPoseHelper.mmpose_to_openpose(kpts2d_mm)
print('kpts2d_mm.shape =', kpts2d_mm.shape)
print('kpts2d_op.shape =', kpts2d_op.shape)
