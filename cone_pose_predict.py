"""Run the trained cone pose model and export pixel boxes, keypoints, and confidences."""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',help='Image, video, or image directory')
    parser.add_argument('--model',type=Path,default=ROOT/'runs/brt-pose-m-960-30k/weights/best.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'pose-predictions')
    parser.add_argument('--confidence',type=float,default=.25)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA GPU unavailable')
    model=YOLO(str(args.model))
    if model.task!='pose':raise ValueError('A pose checkpoint is required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'predictions.jsonl').open('w') as output:
        for index,result in enumerate(model.predict(args.source,stream=True,device=0,imgsz=960,
                                                   conf=args.confidence,verbose=False)):
            boxes=result.boxes.xyxy.cpu().tolist()
            labels=result.boxes.cls.cpu().tolist()
            scores=result.boxes.conf.cpu().tolist()
            points=result.keypoints.data.cpu().tolist()
            payload={'source':result.path,'frame_index':index,'image_height':result.orig_shape[0],
                     'image_width':result.orig_shape[1],'coordinate_units':'original image pixels',
                     'cones':[{'class_id':int(label),'class_name':model.names[int(label)],
                               'confidence':score,'box_xyxy':box,
                               'keypoints_xy_confidence':kp}
                              for label,score,box,kp in zip(labels,scores,boxes,points)]}
            output.write(json.dumps(payload)+'\n')
            result.save(filename=str(args.output/f'{index:06d}-{Path(result.path).stem}.jpg'))
    print(f'Predictions saved to {args.output}')


if __name__=='__main__':main()
