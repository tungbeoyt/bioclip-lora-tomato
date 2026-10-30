"""Verify image pixels against the published fixed split without training."""
from pathlib import Path
import json,hashlib
from collections import Counter
from PIL import Image
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'reference_conservative_d8_v2.json').read_text())
seen=set()
for split,rows in m['splits'].items():
 for row in rows:
  p=root/'data/cls/plantvillage/color'/row['path']
  with Image.open(p) as im:
   rgb=im.convert('RGB');h=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
  if h!=row['pixel_sha256']:raise ValueError('Image changed: '+str(p))
  if h in seen:raise ValueError('Duplicate pixels across manifest records')
  seen.add(h)
 print(split,len(rows),dict(Counter(row['label'] for row in rows)))
print('Verified',len(seen),'unique images; source folder also contains 14 excluded exact duplicates.')
