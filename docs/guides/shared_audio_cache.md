# Audio Cache Backend

Transforms like `AddBackgroundNoise` and `ApplyImpulseResponse` load audio files from disk and resample them on each use. Each transform has an LRU cache to avoid redundant loads within a single process.

However, when processing large datasets with parallel workers, each worker starts with an empty cache. If you have `N` workers processing 10,000 audio files, the same impulse responses and background noises get loaded and resampled `N` times instead of once.

The audio cache backend solves this by providing a shared mutable cache across workers. The example below uses Ray, which allows workers to share a cache actor. The cache populates on-demand as files are loaded (no pre-loading required) and uses LRU eviction when the memory limit is reached.

## Usage with Ray

```python
import numpy as np
import ray
from audiomentations import AddBackgroundNoise, set_audio_cache_backend

@ray.remote
class AudioCacheActor:
    """Shared mutable cache with LRU eviction."""
    
    def __init__(self, max_memory_mb: float = 1000):
        self.cache = {}
        self.max_bytes = int(max_memory_mb * 1024 * 1024)
        self.order = []  # LRU order
    
    def get(self, key: tuple) -> np.ndarray | None:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: tuple, audio: np.ndarray) -> None:
        if key in self.cache:
            return
        # Evict LRU until we have room
        while self._used() + audio.nbytes > self.max_bytes and self.order:
            del self.cache[self.order.pop(0)]
        self.cache[key] = audio
        self.order.append(key)
    
    def _used(self) -> int:
        return sum(v.nbytes for v in self.cache.values())


class RayAudioCacheBackend:
    """Wrapper that adapts the Actor to the audiomentations interface."""
    
    def __init__(self, actor):
        self.actor = actor
    
    def get(self, key: tuple) -> np.ndarray | None:
        return ray.get(self.actor.get.remote(key))
    
    def put(self, key: tuple, audio: np.ndarray) -> None:
        self.actor.put.remote(key, audio)  # fire-and-forget


# Usage
ray.init()
cache_actor = AudioCacheActor.remote(max_memory_mb=500)

@ray.remote
def process_audio(audio, noise_dir, cache_actor):
    set_audio_cache_backend(RayAudioCacheBackend(cache_actor))
    transform = AddBackgroundNoise(
        sounds_path=noise_dir,
        lru_cache_size=0,  # Disable per-transform cache; use shared backend only
        p=1.0
    )
    return transform(audio, sample_rate=44100)

futures = [process_audio.remote(audio, noise_dir, cache_actor) for audio in audio_list]
results = ray.get(futures)
```

## Cache Key Format

```python
cache_key = (file_path, sample_rate, mono)
```

All three must match for a cache hit.

## Cache Behavior

- **Stores full files**: On cache miss, the full file is loaded and cached. Slicing for offset/duration happens on retrieval.
- **LRU eviction**: When memory limit is reached, least recently used items are evicted
- **No pre-loading**: Workers start immediately; cache warms up during processing

## Disable Per-Transform LRU Cache

Transforms like `AddBackgroundNoise` have their own `lru_cache_size` parameter. When using a shared cache backend, you can set this to `0` to avoid double caching:

```python
transform = AddBackgroundNoise(
    sounds_path=noise_dir,
    lru_cache_size=0,  # Rely on shared backend only
    p=1.0
)
```

Without this, audio would be cached twice: once in the shared backend, and again in each transform's per-instance LRU cache.

## Benchmark

With 20 noise files (48kHz → 44.1kHz resampling) and 200 Ray tasks:

```
Without cache: 5.61s
With cache:    0.35s
Speedup:       16.2x

Cache: 20 files, 101 MB, 71.5% hit rate
```

## Backend Interface

Any object with these methods works:

```python
class CacheBackend:
    def get(self, key: tuple) -> np.ndarray | None: ...
    def put(self, key: tuple, audio: np.ndarray) -> None: ...
```

## API

```python
from audiomentations import (
    set_audio_cache_backend,
    get_audio_cache_backend,
    clear_audio_cache_backend,
)
```
