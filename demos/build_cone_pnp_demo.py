from pathlib import Path
import json,base64,cv2,numpy as np
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
model=YOLO(str(ROOT/'runs/brt-pose-m-960-30k/weights/best.pt'))
images=ROOT/'datasets/brt-cone-pose/brt-cone-pose-11k/images/train'
scenes=[]
# Cone keypoint geometry measured from the BRT training labels (median over boxes >=90 px tall):
# pair heights / half-widths as fractions of box height, scaled to FS cone heights.
# Small cones: pairs 0/1, 2/3, 4/5 = top of stripe, bottom of stripe, body foot (above base plate).
# Big orange: pairs 0/1..6/7 = top/bottom edges of the upper and lower white stripes.
def cone_model(h,hw,H):
    return np.array([[s*w*H,-f*H,0] for f,w in zip(h,hw) for s in (-1,1)],np.float64)
OBJ={'small':cone_model([.703,.460,.197],[.122,.166,.230],.325),
     'big':cone_model([.752,.594,.436,.277],[.083,.104,.119,.138],.505)}
for prefix in ['BME_','amz_','gfr_']:
    candidates=sorted(images.glob(prefix+'*'))
    # Choose a training frame with several useful visible cones, without using labels for solves.
    p=candidates[len(candidates)//3]
    result=model.predict(str(p),imgsz=960,device=0,conf=.45,verbose=False)[0]
    h,w=result.orig_shape;f=w/(2*np.tan(np.deg2rad(80)/2));K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
    cones=[]
    for box,kp,cls,score in zip(result.boxes.xyxy.cpu().numpy(),result.keypoints.data.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()):
        cls=int(cls);big=cls==1;H=.505 if big else .325
        obj=OBJ['big' if big else 'small'];n=len(obj)
        use=np.where(kp[:n,2]>=.5)[0]
        solve=None
        if len(use)>=4 and box[3]-box[1]>=12 and cls!=3:
            op=obj[use];ip=kp[use,:2].astype(np.float64)
            try:
                sol=cv2.solvePnPGeneric(op,ip,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
                best=None
                for rv,tv in zip(sol[1],sol[2]):
                    if tv[2,0]<=0:continue
                    pr=cv2.projectPoints(op,rv,tv,K,np.zeros(5))[0].reshape(-1,2)
                    rms=float(np.sqrt(np.mean(np.sum((pr-ip)**2,axis=1))))
                    if best is None or rms<best[0]:best=(rms,rv,tv)
                if best:
                    rms,rv,tv=best
                    if rms<=max(3.,.1*(box[3]-box[1])) and tv[2,0]<80:
                        solve={'t':tv.ravel().tolist(),'R':cv2.Rodrigues(rv)[0].tolist(),'rms':rms,'height':H,'used':use.tolist()}
            except cv2.error:pass
        cones.append({'id':len(cones)+1,'cls':cls,'name':model.names[cls],'confidence':float(score),'box':box.tolist(),'kp':kp.tolist(),'pose':solve})
    im=cv2.imread(str(p));ok,jpg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok
    scenes.append({'name':p.name,'image':'data:image/jpeg;base64,'+base64.b64encode(jpg).decode(),'width':w,'height':h,'fx':f,'cones':cones})
(ROOT/'demos/cone-pnp-data.json').write_text(json.dumps(scenes))
print([(s['name'],len(s['cones']),sum(c['pose'] is not None for c in s['cones'])) for s in scenes])
