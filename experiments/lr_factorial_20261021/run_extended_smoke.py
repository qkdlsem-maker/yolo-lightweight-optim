"""Run the identical frozen native trainer in a separate fixed fixture root."""
from pathlib import Path
import base_train
base_train.R=Path('review_20261021_lr_factorial/extended_smoke')
if __name__=='__main__':base_train.main()
