"""CloudVolume upload helpers for FEABAS rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from feabas import logging

try:  # Optional dependency imported lazily.
    from cloudvolume import CloudVolume
except ImportError:  # pragma: no cover
    CloudVolume = None  # type: ignore


@dataclass
class CloudVolumeParams:
    """Serializable configuration for uploading stitched tiles via CloudVolume."""

    cloudpath: str
    mip: int = 0
    fill_missing: bool = False
    bounded: bool = True
    cache: bool = False
    compress: Optional[str] = None
    parallel: int = 1
    info: Optional[dict] = None
    progress: bool = False
    location: Optional[str] = None
    use_https: bool = True
    background_color: Optional[Iterable[float]] = None
    voxel_offset: Sequence[int] = (0, 0, 0)
    z_start: int = 0
    z_stride: int = 1
    dtype: Optional[str] = None
    num_channels: Optional[int] = None

    def normalized_offset(self) -> np.ndarray:
        return np.asarray(self.voxel_offset, dtype=np.int64)


@dataclass
class CloudVolumeWriter:
    """Thin wrapper that writes FEABAS montage tiles into an existing volume."""

    params: CloudVolumeParams
    _volume: CloudVolume = field(default=None, init=False, repr=False)  # type: ignore
    _dtype: np.dtype = field(default=None, init=False, repr=False)  # type: ignore

    def _ensure_volume(self) -> CloudVolume:
        if CloudVolume is None:
            raise ImportError("cloudvolume must be installed to enable CloudVolume rendering.")
        if self._volume is None:
            self._volume = CloudVolume(
                self.params.cloudpath,
                mip=self.params.mip,
                fill_missing=self.params.fill_missing,
                bounded=self.params.bounded,
                cache=self.params.cache,
                compress=self.params.compress,
                info=self.params.info,
                progress=self.params.progress,
                parallel=self.params.parallel,
                location=self.params.location,
                use_https=self.params.use_https,
                background_color=self.params.background_color,
            )
            self._dtype = np.dtype(self.params.dtype or self._volume.dtype)
            if (self.params.num_channels is None) and hasattr(self._volume, "num_channels"):
                self.params.num_channels = int(self._volume.num_channels)
        return self._volume

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._ensure_volume()
        return self._dtype

    @property
    def num_channels(self) -> int:
        if self.params.num_channels is None:
            self._ensure_volume()
        return int(self.params.num_channels or 1)

    def tile_bbox_to_volume(self, bbox: Sequence[int], z_index: int) -> Tuple[int, int, int, int, int, int]:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        offset = self.params.normalized_offset()
        z_voxel = self.params.z_start + z_index * self.params.z_stride
        return x0 + offset[0], y0 + offset[1], z_voxel, x1 + offset[0], y1 + offset[1], z_voxel + 1

    def write_tile(self, bbox: Sequence[int], image: np.ndarray, z_index: int) -> bool:
        vol = self._ensure_volume()
        xmin, ymin, zmin, xmax, ymax, zmax = self.tile_bbox_to_volume(bbox, z_index)
        if image is None:
            return False
        if image.ndim == 2:
            tile = image[:, :, np.newaxis]
        else:
            tile = image
        if tile.shape[-1] == 1 and self.num_channels == 1:
            pass
        elif tile.shape[-1] != self.num_channels:
            raise ValueError(
                f"Tile channel count {tile.shape[-1]} does not match CloudVolume channels {self.num_channels}."
            )
        tile = np.asarray(tile.swapaxes(0, 1), dtype=self.dtype, order="C")
        if tile.size == 0:
            return False
        vol[xmin:xmax, ymin:ymax, zmin:zmax] = tile
        return True

    def flush_info(self) -> None:
        if self._volume is None:
            return
        logger = logging.get_logger(None)
        try:
            self._volume.refresh_info()
        except Exception as exc:  # pragma: no cover
            if logger:
                logger.warning(f"CloudVolume info refresh failed: {exc}")


def build_writer(params_dict: dict | CloudVolumeParams) -> CloudVolumeWriter:
    """Create a writer from configuration or an existing dataclass."""

    if isinstance(params_dict, CloudVolumeParams):
        params = params_dict
    else:
        params = CloudVolumeParams(**params_dict)
    return CloudVolumeWriter(params)
