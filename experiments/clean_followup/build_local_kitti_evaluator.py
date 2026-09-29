"""Adapt the checksum-pinned official 2D scoring core for local paths, no mail.

Matching, ignored regions, thresholds, difficulty and precision-envelope code
are unchanged. No third-party reimplementation substitutes for the native core.
"""
import argparse,difflib,hashlib,json,subprocess,zipfile
from pathlib import Path

MAIN=r'''
// Local harness: no email, shell commands, plotting, 3D or leaderboard actions.
int main(int argc,char* argv[]) {
  if(argc!=5) { std::cerr << "Usage: eval_2d GT_DIR DET_DIR OUT_DIR N_IMAGES\n"; return 2; }
  const std::string gt_dir=argv[1], det_dir=argv[2], out_dir=argv[3];
  const int count=std::stoi(argv[4]);
  if(count<=0) return 2;
  initGlobals();
  std::vector<std::vector<tGroundtruth>> gt;
  std::vector<std::vector<tDetection>> det;
  bool aos=false;
  std::vector<bool> ei(NUM_CLASS,false),eg(NUM_CLASS,false),e3(NUM_CLASS,false);
  for(int i=0;i<count;++i) {
    char name[32]; std::snprintf(name,sizeof(name),"%06d.txt",i);
    bool a,b;
    gt.push_back(loadGroundtruth(gt_dir+"/"+name,a));
    det.push_back(loadDetections(det_dir+"/"+name,aos,ei,eg,e3,b));
    if(!a || !b) { std::cerr << "Missing label/prediction " << name << std::endl; return 3; }
  }
  // Both prespecified classes are always evaluated; absent predictions score 0.
  for(int c=0;c<2;++c) {
    FILE* fp=std::fopen((out_dir+"/stats_"+CLASS_NAMES[c]+"_detection.txt").c_str(),"w");
    if(!fp) return 4;
    for(int d=0;d<3;++d) {
      std::vector<double> precision,orientation;
      if(!eval_class(fp,nullptr,(CLASSES)c,gt,det,false,imageBoxOverlap,precision,orientation,(DIFFICULTY)d,IMAGE)) return 5;
    }
    std::fclose(fp);
  }
  return 0;
}
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='review_20260929_clean/external_kitti');a=p.parse_args();root=Path(a.root)
    archive=root/'devkit_object.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1'
    with zipfile.ZipFile(archive) as z:
        member=[n for n in z.namelist() if n.endswith('/cpp/evaluate_object.cpp') or n=='cpp/evaluate_object.cpp'];assert len(member)==1
        raw=z.read(member[0]);original=raw.decode()
    out=root/'local_evaluator';out.mkdir(exist_ok=True)
    (out/'evaluate_object_original.cpp').write_bytes(raw)
    code=original[:original.index('void saveAndPlotPlots(')]
    begin=code.index('#include <boost/numeric/ublas/matrix.hpp>');end=code.index('using namespace std;')
    code=code[:begin]+'#include <string>\n\n'+code[end:]
    begin=code.index('// compute polygon of an oriented bounding box');end=code.index('vector<double> getThresholds(')
    code=code[:begin]+code[end:]
    code=code.replace('const int32_t N_TESTIMAGES = 7518;','// Frame count is supplied to the local harness.')
    # Higher output precision changes only serialization, never matching/AP.
    code=code.replace('fprintf(fp_det,"%f ",precision[i]);','fprintf(fp_det,"%.17g ",precision[i]);')
    code+=MAIN
    for start,end in [('vector<double> getThresholds(', 'void cleanData('),('void cleanData(', 'tPrData computeStatistics('),('tPrData computeStatistics(', 'bool eval_class ('),('bool eval_class (', 'void saveAndPlotPlots(')]:
        block=original[original.index(start):original.index(end)]
        assert block in code,'Scoring core unexpectedly modified: '+start
    target=out/'evaluate_2d_local.cpp';target.write_text(code)
    patch=''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='evaluate_object_original.cpp',tofile='evaluate_2d_local.cpp'))
    (out/'local_harness.patch').write_text(patch)
    command=['g++','-O2','-std=c++11',str(target),'-o',str(out/'evaluate_2d_local')]
    subprocess.run(command,check=True)
    manifest={'upstream_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'upstream_source_sha256':hashlib.sha256(raw).hexdigest(),'adapted_source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'executable_sha256':hashlib.sha256((out/'evaluate_2d_local').read_bytes()).hexdigest(),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0], 'command':command,'matching_core_unchanged':True,'changes':['local explicit paths and frame count','Car and Pedestrian always evaluated, including zero predictions','2D only; remove unused Boost 3D routines and leaderboard/mail/plot harness','17-digit precision serialization','R40 is computed externally as mean of precision slots 1 through 40']}
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('OFFICIAL_2D_CORE_COMPILED',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
