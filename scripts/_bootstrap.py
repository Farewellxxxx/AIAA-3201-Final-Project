"""Use project-local extra libraries and writable caches."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('PYTHONIOENCODING','utf-8')
for p in ['.cache/matplotlib','.cache/torch','.cache/huggingface','.cache/tmp','results']:
    (ROOT/p).mkdir(parents=True,exist_ok=True)
import tempfile
tempfile.tempdir=str(ROOT/'.cache/tmp')

def add_repo(name): sys.path.insert(0,str(ROOT/'third_party'/name))
