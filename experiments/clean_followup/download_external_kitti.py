"""Download checksum-pinned KITTI inputs and extract only labeled-partition images."""
import hashlib,json,time,urllib.request,zipfile
from pathlib import Path
R=Path('review_20260929_clean/external_kitti');R.mkdir(parents=True,exist_ok=True)
archives=[('devkit_object.zip',63794,'ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1'),('data_object_label_2.zip',5601213,'4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880'),('data_object_image_2.zip',12569945557,'351c5a2aa0cd9238b50174a3a62b846bc5855da256b82a196431d60ff8d43617')]
records=[]
for name,size,digest in archives:
    url='https://s3.eu-central-1.amazonaws.com/avg-kitti/'+name;p=R/name;partial=R/(name+'.partial')
    if not p.exists():
        offset=partial.stat().st_size if partial.exists() else 0
        req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(req,timeout=60) as response:
            if offset:assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {offset}-')
            with partial.open('ab' if offset else 'wb') as f:
                latest=time.time()
                while block:=response.read(4*1024*1024):
                    f.write(block)
                    if time.time()-latest>30:print('DOWNLOAD',name,f.tell(),'/',size,flush=True);latest=time.time()
        assert partial.stat().st_size==size
        partial.replace(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    assert p.stat().st_size==size and h.hexdigest()==digest,(name,'checksum mismatch')
    record={'url':url,'filename':name,'bytes':size,'sha256':digest,'checksum_registry':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/kitti/checksums.tsv'}
    with zipfile.ZipFile(p) as z:
        members=[n for n in z.namelist() if not n.endswith('/') and (name=='devkit_object.zip' or n.startswith('training/'))]
        for n in members:
            base=R/'devkit' if name=='devkit_object.zip' else R/'data'
            out=(base/n).resolve();assert out.is_relative_to(base.resolve());out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():out.write_bytes(z.read(n))
            else:assert out.stat().st_size==z.getinfo(n).file_size
        record['extracted_files']=len(members)
    records.append(record);(R/'download_manifest.json').write_text(json.dumps(records,indent=2));print('VERIFIED',name,flush=True)
assert len(list((R/'data/training/image_2').glob('*.png')))==7481
assert len(list((R/'data/training/label_2').glob('*.txt')))==7481
(R/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'status':'complete','labeled_images':7481,'archives':records},indent=2))
print('KITTI_INPUTS_COMPLETE',flush=True)
