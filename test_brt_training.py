import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from augment_brt_pose import geometry
from train_brt_full import KeypointTrainer
from evaluate_brt_pose import evaluate


class TrainingChecks(unittest.TestCase):
    def test_pixels_and_keypoints_share_geometry(self):
        image=np.zeros((300,400,3),dtype=np.uint8)
        kp=np.array([[.45,.4,2],[.55,.4,2],[.45,.5,2],[.55,.5,2],
                     [.45,.6,2],[.55,.6,2],[0,0,0],[0,0,0]])
        for x,y,v in kp:
            if v:cv2.circle(image,(round(x*400),round(y*300)),4,(255,255,255),-1)
        rows=np.r_[0,.5,.5,.3,.4,kp.ravel()][None,:]
        transformed,labels,_=geometry(image,rows,np.random.default_rng(20261010))
        self.assertEqual(labels.shape,(1,29))
        new_kp=labels[0,5:].reshape(8,3)
        np.testing.assert_array_equal(new_kp[6:],0)
        for x,y,v in new_kp[:6]:
            self.assertEqual(v,2)
            x,y=round(x*400),round(y*300)
            self.assertGreater(transformed[y-2:y+3,x-2:x+3].max(),180)

    def test_background_images_keep_empty_labels(self):
        _,rows,_=geometry(np.zeros((100,100,3),np.uint8),np.empty((0,29)),np.random.default_rng(1))
        self.assertEqual(rows.shape,(0,29))

    def test_checkpoint_selection_uses_pose_only(self):
        trainer=SimpleNamespace(best_fitness=.5)
        def default_validate(obj):
            obj.best_fitness=1.7
            return {'metrics/mAP50-95(P)':.6},1.7
        with patch('ultralytics.models.yolo.pose.train.PoseTrainer.validate',default_validate):
            # Bind to a real subclass instance without initializing GPU training.
            trainer=object.__new__(KeypointTrainer);trainer.best_fitness=.5
            _,fitness=trainer.validate()
        self.assertEqual(fitness,.6)
        self.assertEqual(trainer.best_fitness,.6)

    def test_missed_cones_reduce_end_to_end_keypoint_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'images/test').mkdir(parents=True)
            (root/'labels/test').mkdir(parents=True)
            kp=np.tile([.25,.25,2],(8,1))
            rows=np.array([np.r_[0,.25,.25,.2,.2,kp.ravel()],
                           np.r_[0,.75,.75,.2,.2,np.tile([.75,.75,2],8)]])
            np.savetxt(root/'labels/test/example.txt',rows)
            config=root/'data.yaml';config.write_text(yaml.safe_dump({'path':str(root),'test':str(root/'images/test')}))
            prediction=SimpleNamespace(
                path=str(root/'images/test/example.jpg'),orig_shape=(100,100),
                boxes=SimpleNamespace(xyxy=torch.tensor([[15,15,35,35]]),cls=torch.tensor([0.])),
                keypoints=SimpleNamespace(data=torch.tensor([[[25,25,1.]]*8])),
                plot=lambda:np.zeros((100,100,3),np.uint8))
            class Boxes(SimpleNamespace):
                def __len__(self):return len(self.xyxy)
            prediction.boxes=Boxes(**vars(prediction.boxes))
            model=SimpleNamespace(predict=lambda **kwargs:iter([prediction]))
            result=evaluate(model,config,root/'report.json')
            self.assertEqual(result['matched_PCK_05'],1)
            self.assertEqual(result['end_to_end_PCK_05'],.5)
            self.assertEqual(result['cone_recall'],.5)


if __name__=='__main__':unittest.main()
