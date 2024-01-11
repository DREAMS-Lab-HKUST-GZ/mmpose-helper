- MMPose with RTMW-x
  - works with `dev-1.x` https://github.com/open-mmlab/mmpose/tree/dev-1.x
  - probability also works for v1.3.0 https://github.com/open-mmlab/mmpose/releases/tag/v1.3.0

-----
- for demo run `python t_mmpose_helper.py`
  - mmpose-style skeleton has 133 keypoints
  - use `MMPoseHelper.mmpose_to_openpose` to convert to openpose-style 118 keypoints (cooresponding to `bodyhandface` of [EasyMocap](https://github.com/zju3dv/EasyMocap))
    - from `SMPL-X` to openpose-style 118 keypoints: [J_regressor_body25_smplx.txt](https://github.com/zju3dv/EasyMocap/blob/master/data/smplx/J_regressor_body25_smplx.txt)
      - demo usage: [X_regressor](https://github.com/zju3dv/EasyMocap/blob/master/easymocap/smplmodel/body_model.py#L133C13-L139)
-----
![example](data/test01.mmpose.png)
