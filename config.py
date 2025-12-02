import os
from pathlib import Path

DEBUG = True
RERUN = False
YASH = False

HOME_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
print(HOME_DIR)

# Environment setup
if (env_path := HOME_DIR / ".env").exists():
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

CACHE_DIR = HOME_DIR / 'cache'
DEBUG_DIR = CACHE_DIR / 'debug'
PLOTS_DIR = CACHE_DIR / 'plots'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["RESULTCACHING_HOME"] = str(CACHE_DIR / "resultcaching")
os.environ['MMAP_HOME'] = str(CACHE_DIR / 'mmap')
os.environ["BRAINIO_HOME"] = str(CACHE_DIR / "brainio2")
os.environ["BRAINSCORE_HOME"] = str(CACHE_DIR / "brain-score")
os.environ['TORCH_HOME'] = str(CACHE_DIR / 'torch')
os.environ['HF_HOME'] = str(CACHE_DIR / 'hf')

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["RESULTCACHING_DISABLE"] = '0'

if DEBUG:
    print("*" * 100)
    print(" " * 30 + f"WARNING: DEBUG MODE IS ON, RERUN {'ON' if RERUN else 'OFF'}")
    print("*" * 100)


# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'
# rcParams['mathtext.fontset'] = 'dejavusans'


ROOT_KINETICS400 = '/mnt/scratch/fkolly/datasets/kinetics-dataset/k400'
ROOT_IMAGENETVID = '/mnt/scratch/akgokce/datasets/imagenet'
ROOT_AFD101 = '/mnt/scratch/fkolly/datasets/AFD101'
ROOT_SSV2 = '/mnt/scratch/fkolly/datasets/smthsmthv2'
ROOT_AFRAZ2006 = '/mnt/scratch/ytang/datasets/afraz2006'

PRETRAINED_DIR = "/mnt/scratch/fkolly/brainmo/pretrained"
POSITION_DIR = CACHE_DIR / "positions"
POSITION_DIR.mkdir(exist_ok=True, parents=True)


if YASH:
    ROOT_KINETICS400 = '/data2/ynshah/Kinetics400/k400'
    ROOT_IMAGENETVID = '/data2/ynshah/imagenet-vid'
    PRETRAINED_DIR = "/data2/ynshah/tdann-transform/cache/checkpoints"
    from spacetorch.paths import POSITION_DIR
