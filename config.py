import os
from pathlib import Path

HOME_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Environment setup
if (env_path := HOME_DIR / ".env").exists():
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

CACHE_DIR = HOME_DIR / 'cache'
FIGURE_DIR = HOME_DIR / 'figures'

os.environ["RESULTCACHING_HOME"] = str(CACHE_DIR / "resultcaching")
os.environ['MMAP_HOME'] = str(CACHE_DIR / 'mmap')
os.environ["BRAINIO_HOME"] = str(CACHE_DIR / "brainio2")
os.environ["BRAINSCORE_HOME"] = str(CACHE_DIR / "brain-score")
os.environ['TORCH_HOME'] = str(CACHE_DIR / 'torch')
os.environ['HF_HOME'] = str(CACHE_DIR / 'hf')

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["RESULTCACHING_DISABLE"] = '0'