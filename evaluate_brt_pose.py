"""Measure strict normalized keypoint error, including missed cones, on held-out data."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml


def evaluate(model,data_yaml,output,imgsz=960,split='test'):
    config=yaml.safe_load(Path(data_yaml).read_text())
    directory=Path(config[split])
    label_dir=Path(config['path'])/'labels'/split
    totals=np.zeros(8,dtype=int)
    correct05=np.zeros(8,dtype=int)
    correct10=np.zeros(8,dtype=int)
    matched_totals=np.zeros(8,dtype=int)
    errors=[];pixels=[];gt_objects=0;matched_objects=0;predicted_objects=0
    predictions=model.predict(source=str(directory),stream=True,device=0,imgsz=imgsz,
                              conf=0.25,iou=0.7,max_det=300,verbose=False)
    examples=Path(output).parent/f'{split}-examples';examples.mkdir(exist_ok=True)
    for index,result in enumerate(predictions):
        h,w=result.orig_shape
        label=label_dir/f'{Path(result.path).stem}.txt'
        rows=np.asarray([list(map(float,line.split())) for line in label.read_text().splitlines()
                         if line.strip()]).reshape(-1,29)
        gt_objects+=len(rows);predicted_objects+=len(result.boxes)
        gt_kp=rows[:,5:].reshape(-1,8,3)
        totals+=(gt_kp[:,:,2]>0).sum(0)
        centers=rows[:,1:3]*[w,h];sizes=rows[:,3:5]*[w,h]
        gt_boxes=np.c_[centers-sizes/2,centers+sizes/2]
        boxes=result.boxes.xyxy.cpu().numpy()
        cls=result.boxes.cls.cpu().numpy()
        if len(boxes) and len(rows):
            lo=np.maximum(gt_boxes[:,None,:2],boxes[None,:,:2])
            hi=np.minimum(gt_boxes[:,None,2:],boxes[None,:,2:])
            intersection=np.maximum(hi-lo,0).prod(-1)
            area_gt=np.maximum(gt_boxes[:,2:]-gt_boxes[:,:2],0).prod(-1)
            area_pr=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(-1)
            iou=intersection/(area_gt[:,None]+area_pr[None,:]-intersection+1e-9)
            iou*=rows[:,0,None]==cls[None,:]
            g,p=linear_sum_assignment(-iou)
            pred_kp=result.keypoints.data.cpu().numpy()
            for gi,pi in zip(g,p):
                if iou[gi,pi]<0.5:continue
                matched_objects+=1
                visible=gt_kp[gi,:,2]>0
                distance=np.linalg.norm(pred_kp[pi,:,:2]-gt_kp[gi,:,:2]*[w,h],axis=1)
                normalized=distance/max(sizes[gi,1],1)
                # Count a point as successful only when both location and visibility agree.
                confident=pred_kp[pi,:,2]>=0.5
                matched_totals+=visible
                correct05+=visible & confident & (normalized<=0.05)
                correct10+=visible & confident & (normalized<=0.10)
                errors.extend(normalized[visible].tolist());pixels.extend(distance[visible].tolist())
        if index%100==0:
            print(f'Pose error evaluation {index+1} images',flush=True)
            cv2.imwrite(str(examples/f'{Path(result.path).stem}.jpg'),result.plot())
    report={
        'split':split,
        'confidence_threshold':0.25,'keypoint_confidence_threshold':0.5,
        'matching':'same-class one-to-one box IoU >= 0.5',
        'normalization':'ground-truth bounding box height',
        'ground_truth_cones':gt_objects,'predicted_cones':predicted_objects,'matched_cones':matched_objects,
        'cone_recall':matched_objects/max(gt_objects,1),'cone_precision':matched_objects/max(predicted_objects,1),
        'end_to_end_PCK_05':float(correct05.sum()/max(totals.sum(),1)),
        'end_to_end_PCK_10':float(correct10.sum()/max(totals.sum(),1)),
        'matched_PCK_05':float(correct05.sum()/max(matched_totals.sum(),1)),
        'matched_PCK_10':float(correct10.sum()/max(matched_totals.sum(),1)),
        'median_error_box_heights':float(np.median(errors)) if errors else None,
        'median_error_pixels':float(np.median(pixels)) if pixels else None,
        'per_keypoint':[{'index':k,'labeled':int(totals[k]),'matched':int(matched_totals[k]),
                         'end_to_end_PCK_05':float(correct05[k]/max(totals[k],1)),
                         'end_to_end_PCK_10':float(correct10[k]/max(totals[k],1))} for k in range(8)]}
    Path(output).write_text(json.dumps(report,indent=2))
    return report
