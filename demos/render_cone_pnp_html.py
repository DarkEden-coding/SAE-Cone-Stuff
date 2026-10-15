"""Embed cone-pnp-data.json (with downscaled frames) into cone-pnp-template.html."""
from pathlib import Path
import base64,json,cv2,numpy as np
HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'cone-pnp-data.json').read_text())
for s in data:
    im=cv2.imdecode(np.frombuffer(base64.b64decode(s['image'].split(',')[1]),np.uint8),1)
    W=min(2400,im.shape[1]);im=cv2.resize(im,(W,round(im.shape[0]*W/im.shape[1])),interpolation=cv2.INTER_AREA)  # near full res so zooming stays sharp
    s['image']='data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,84])[1]).decode()
html=(HERE/'cone-pnp-template.html').read_text().replace('__DATA__',json.dumps(data))
(HERE/'cone-pnp-demo.html').write_text(html);print('HTML bytes',len(html))
