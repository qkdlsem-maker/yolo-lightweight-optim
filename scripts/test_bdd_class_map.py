"""The official-category converter must agree with the training configuration."""
from pathlib import Path
import argparse
import yaml
from prepare_bdd100k import CLASS_MAP

p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/bdd100k.yaml'));args=p.parse_args()
config=yaml.safe_load(Path(args.config).read_text())
names=config['names']
aliases={'pedestrian':'person','motorcycle':'motor','bicycle':'bike'}
assert len(CLASS_MAP)==len(names)==10
for category,index in CLASS_MAP.items():
    name=names[index] if isinstance(names,list) else names[index]
    assert name==aliases.get(category,category),(category,index,name)
print('PASS: official detection categories map to the configured ten class names')
