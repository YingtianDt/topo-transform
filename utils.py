from config import DEBUG_DIR, DEBUG, RERUN

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional
from functools import wraps


def cached(
    cache_name: str,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
    persistent: bool = False,
    rerun: bool = False,
):
    """Decorator for caching function results.
    
    Args:
        cache_name: Name identifier for this cache
        cache_dir: Optional custom cache directory
        verbose: Print cache hit/miss messages
        
    Usage:
        @cached('my_computation')
        def expensive_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If DEBUG is False, always recompute
            if not DEBUG and not persistent:
                if verbose:
                    print(f"[Cache] DEBUG=False, computing {cache_name}")
                return func(*args, **kwargs)
            
            # Setup cache directory
            _cache_dir = cache_dir or DEBUG_DIR
            _cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache file path
            cache_file = _cache_dir / f"{cache_name}.pkl"
            
            # If RERUN is True, force recompute
            if RERUN or rerun:
                if verbose:
                    print(f"[Cache] RERUN=True, recomputing {cache_name}")
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                if verbose:
                    print(f"[Cache] Saved to {cache_file}")
                return result
            
            # Try to load from cache
            if cache_file.exists():
                if verbose:
                    print(f"[Cache] Loading {cache_name} from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Cache miss - compute and store
            if verbose:
                print(f"[Cache] Cache miss, computing {cache_name}")
            result = func(*args, **kwargs)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            if verbose:
                print(f"[Cache] Saved to {cache_file}")
            
            return result
        
        return wrapper
    return decorator
