import os
from pathlib import Path

HOME_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

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

DEBUG = False
RERUN = not DEBUG

if DEBUG:
    print("*" * 100)
    print(" " * 35 + "WARNING: DEBUG MODE IS ON.")
    print("*" * 100)


# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'
# rcParams['mathtext.fontset'] = 'dejavusans'