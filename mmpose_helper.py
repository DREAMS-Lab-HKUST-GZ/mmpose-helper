# MMPose Helper.
# Contributer(s): Neil Z. Shao
# HKUST(GZ) 2023.
import os
import numpy as np
from mmpose.apis import inference_topdown, init_model, inference_bottomup
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances

def get_model_path(fn=None):
    dir = os.path.abspath(os.path.join(__file__, '../models'))
    if fn is None:
        return dir
    else:
        return os.path.join(dir, fn)  

class MMPoseHelper():
    """ code altered from topdown_demo_with_mmdet
      - https://github.com/open-mmlab/mmpose/blob/main/demo/topdown_demo_with_mmdet.py
      - to use rtmw-x:
        - fix by https://github.com/open-mmlab/mmpose/issues/2741#issuecomment-1761314968
        - https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
      - mmpose checkout to `dev-1.x`
        - https://github.com/open-mmlab/mmpose/tree/dev-1.x
      - but use config from `main`
        - https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-x_8xb320-270e_cocktail13-384x288.py
      - ckpt are the same for `main` and `dev-1.x`
        - https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw_cocktail13.md
    """
    def __init__(self, skeleton_style='mmpose',
                 device='cuda') -> None:
        pose2d = get_model_path('rtmw-x_8xb320-270e_cocktail13-384x288.py')
        pose2d_weights = get_model_path('rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.pth')
        self.pose_estimator = init_model(pose2d, pose2d_weights, device=device)

        det_config = get_model_path('rtmdet_m_8xb32-300e_coco.py')
        det_checkpoint = get_model_path('rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth')
        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.det_cat_id = [0]
        self.bbox_thr = 0.3
        self.nms_thr = 0.3

        self.skeleton_style = skeleton_style
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style=self.skeleton_style)
        self.draw_heatmap = False
        self.draw_bbox = True
        self.show_kpt_idx = True
        self.show = False
        self.show_interval = 0
        self.kpt_thr = 0.3

    def process_one_image(self, img_path):
        # predict bbox
        det_result = inference_detector(self.detector, img_path)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id, pred_instance.scores > self.bbox_thr)]
        bboxes = bboxes[nms(bboxes, self.nms_thr), :4]

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img_path, bboxes)
        data_samples = merge_data_samples(pose_results)
        mmpose_out = {
            'data_samples': data_samples,
        }

        pred_instances = data_samples.get('pred_instances', None)
        if pred_instances is not None:
            pred_instances = split_instances(pred_instances)
            for inst in pred_instances:
                for key in inst:
                    if isinstance(inst[key], np.float32):
                        inst[key] = inst[key].item()
            mmpose_out['pred_instances'] = pred_instances
        else:
            mmpose_out['pred_instances'] = []

        return mmpose_out
    
    def visualize(self, img, mmpose_out):
        data_samples = mmpose_out['data_samples']

        self.visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=self.draw_heatmap,
            draw_bbox=self.draw_bbox,
            show_kpt_idx=self.show_kpt_idx,
            skeleton_style=self.skeleton_style,
            show=self.show,
            wait_time=self.show_interval,
            kpt_thr=self.kpt_thr)
        return self.visualizer.get_image()
        
    ##########
    # mmpose 133 keypoints to openpose 118 keypoints
    @staticmethod
    def mmpose_to_openpose(kpts_mm):
        # NO_POINTS = np.zeros_like(kpts2d[:, :1, :])
        if len(kpts_mm.shape) == 2:
            kpts_mm = kpts_mm[None, ...]
            in_dims = 2
        else:
            kpts_mm = kpts_mm
            in_dims = 3

        kpts_op = np.concatenate([
            kpts_mm[:, [0], :],                                  # 0
            kpts_mm[:, [5, 6], :].mean(axis=1, keepdims=True),   # 1
            kpts_mm[:, [6, 8, 10, 5, 7, 9], :],                  # 2, 3, 4, 5, 6, 7
            kpts_mm[:, [11, 12], :].mean(axis=1, keepdims=True), # 8
            kpts_mm[:, [12, 14, 16, 11, 13, 15], :],             # 9, 10, 11, 12, 13, 14
            kpts_mm[:, [2, 1, 4, 3], :],                         # 15, 16, 17, 18
            kpts_mm[:, [17, 18, 19, 20, 21, 22], :],             # 19, 20, 21, 22, 23, 24
            kpts_mm[:, 91:112, :],                               # [25-45]
            kpts_mm[:, 112:133, :],                              # [46-66]
            kpts_mm[:, 40:91, :],                                # [67-117]
        ], axis=1)

        if in_dims == 2:
            kpts_op = kpts_op[0]
        return kpts_op

        