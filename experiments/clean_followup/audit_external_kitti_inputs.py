"""Verify extracted KITTI inputs and compare encoded image hashes with BDD."""
import json,zipfile,zlib,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from clean_followup_utils import sha,dump
R=Path('review_20260929_clean/external_kitti');data=R/'data/training'
assert json.loads((R/'DOWNLOAD_COMPLETE.json').read_text())['labeled_images']==7481
records=[];dimensions=Counter();classes=Counter();label_rows=0
for archive,folder,suffix in [('data_object_image_2.zip','image_2','.png'),('data_object_label_2.zip','label_2','.txt')]:
    with zipfile.ZipFile(R/archive) as z:
        for i in range(7481):
            name=f'{i:06d}'+suffix;p=data/folder/name;raw=p.read_bytes()
            member=z.getinfo('training/'+folder+'/'+name)
            assert len(raw)==member.file_size and (zlib.crc32(raw)&0xffffffff)==member.CRC,p
            if suffix=='.png':
                with Image.open(p) as im:dimensions[str(im.size)]+=1
                records.append({'image_id':p.stem,'sha256':sha(p),'bytes':len(raw)})
            else:
                for line in raw.decode().splitlines():
                    tokens=line.split();assert len(tokens)==15
                    classes[tokens[0]]+=1;label_rows+=1
    print('EXTRACTED_CRC_VERIFIED',archive,flush=True)
prior=Path('review_20260929/images_sha256_manifest.jsonl')
bdd=[json.loads(line) for line in prior.read_text().splitlines()];assert len(bdd)==100000
hashes={r['sha256'] for r in bdd};overlap=[r for r in records if r['sha256'] in hashes]
assert not overlap,'Unexpected BDD/KITTI exact encoded overlap; stop before inference'
def size(row):
    p=Path('data/raw')/row['split']/'images'/row['name']
    assert p.stat().st_size==row['bytes'],'BDD file size changed from preceding hash audit'
    with Image.open(p) as im:return str(im.size)
with ThreadPoolExecutor(max_workers=4) as pool:bdd_dimensions=Counter(pool.map(size,bdd))
shared_dimensions=sorted(set(bdd_dimensions)&set(dimensions))
out={'status':'passed','kitti_images':7481,'kitti_label_rows':label_rows,'label_class_counts':dict(classes),'extracted_archive_CRC_checks_passed':True,'kitti_image_dimensions':dict(dimensions),'bdd_image_dimensions':dict(bdd_dimensions),'bdd_encoded_hash_manifest_sha256':sha(prior),'bdd_hashes_from_prior_recorded_immutable_data_audit':True,'bdd_current_file_sizes_match_prior_manifest':True,'encoded_file_overlap':overlap,'common_image_dimensions':shared_dimensions,'exact_native_pixel_array_equality_ruled_out_by_disjoint_dimensions':not shared_dimensions,'scope':'Exact encoded-file hashes and native image dimensions. Does not establish perceptual/sequence independence, exclude crops/resizing, or audit COCO pretraining image overlap. KITTI public training partition is external only relative to this study.'}
dump(R/'image_manifest.json',records);dump(R/'input_audit.json',out);print('KITTI_INPUT_AUDIT_PASSED',flush=True)
