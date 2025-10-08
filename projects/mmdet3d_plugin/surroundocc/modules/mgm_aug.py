import numpy as np
import cv2
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class MultiViewGridMask:
    """
    Multi-view GridMask (MGM) augmentation for surround-view images.

    Implements:
    - Dynamic grid adjustment based on object scale
    - Edge dilution for FOV boundary regions
    - Epoch-dependent probability scheduling
    - YOLOv8-based small object detection

    Args:
        use_h (bool): Apply horizontal grid masks
        use_w (bool): Apply vertical grid masks
        rotate (int): Max rotation angle (not used in MGM)
        offset (bool): Random offset for grid alignment
        ratio_range (tuple): Range for mask ratio (r_h, r_w)
        density_range (tuple): Range for grid density (d_h, d_w)
        mode (int): 0=mask pixels to 0, 1=keep pixels
        prob_schedule (str): 'dynamic' or 'fixed'
        edge_margin (int): Pixel margin for edge dilution (default: 50)
        small_obj_threshold (float): Object size threshold (default: 0.1)
        large_mask_ratio (float): Large mask size ratio (default: 0.05 = 1/20)
        detector_path (str): Path to YOLOv8 weights
        cache_dir (str): Directory for caching detection results
    """

    def __init__(
            self,
            use_h: bool = True,
            use_w: bool = True,
            rotate: int = 1,
            offset: bool = False,
            ratio_range: Tuple[float, float] = (0.5, 0.8),
            density_range: Tuple[float, float] = (0.4, 0.7),
            mode: int = 1,
            prob_schedule: str = 'dynamic',
            edge_margin: int = 50,
            small_obj_threshold: float = 0.1,
            large_mask_ratio: float = 0.05,
            detector_path: Optional[str] = None,
            cache_dir: str = './work_dirs/mgm_cache'
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio_range = ratio_range
        self.density_range = density_range
        self.mode = mode
        self.prob_schedule = prob_schedule
        self.edge_margin = edge_margin
        self.small_obj_threshold = small_obj_threshold
        self.large_mask_ratio = large_mask_ratio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Current epoch (will be updated externally)
        self.current_epoch = 0

        # Initialize detector
        self.detector = None
        if detector_path and YOLO_AVAILABLE:
            try:
                self.detector = YOLO(detector_path)
                print(f"MGM: Loaded YOLOv8 detector from {detector_path}")
            except Exception as e:
                print(f"MGM: Failed to load detector: {e}")

        # Detection cache
        self.detection_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached detection results"""
        cache_file = self.cache_dir / 'detection_cache.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.detection_cache = pickle.load(f)
                print(f"MGM: Loaded {len(self.detection_cache)} cached detections")
            except Exception as e:
                print(f"MGM: Failed to load cache: {e}")

    def _save_cache(self):
        """Save detection results to cache"""
        cache_file = self.cache_dir / 'detection_cache.pkl'
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.detection_cache, f)
        except Exception as e:
            print(f"MGM: Failed to save cache: {e}")

    def _detect_small_objects(self, img: np.ndarray, img_id: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect small objects using YOLOv8

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Check cache first
        if img_id in self.detection_cache:
            return self.detection_cache[img_id]

        # Run detection if detector available
        if self.detector is None:
            return []

        try:
            # Run YOLOv8 detection (conf=0.25, iou=0.45 as per paper)
            results = self.detector(img, conf=0.25, iou=0.45, verbose=False)

            boxes = []
            h, w = img.shape[:2]
            threshold_size = min(h, w) * self.small_obj_threshold  # 1/10 of shorter dimension

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_w, box_h = x2 - x1, y2 - y1

                    # Filter small objects
                    if box_w < threshold_size and box_h < threshold_size:
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))

            # Cache results
            self.detection_cache[img_id] = boxes

            return boxes

        except Exception as e:
            print(f"MGM: Detection failed: {e}")
            return []

    def _get_augmentation_probability(self) -> float:
        """
        Calculate MGM probability based on current epoch
        p_MGM(e) = min(0.5, 0.3 + 0.05 × ⌊e/10⌋)
        """
        if self.prob_schedule == 'fixed':
            return 0.5

        # Dynamic schedule as per paper
        prob = 0.3 + 0.05 * (self.current_epoch // 10)
        return min(0.5, prob)

    def _create_grid_mask(self, h: int, w: int, grid_size: int, ratio: float) -> np.ndarray:
        """Create basic grid mask pattern"""
        mask = np.ones((h, w), dtype=np.uint8)

        mask_size = int(grid_size * ratio)

        if self.use_h:
            for i in range(0, h, grid_size):
                mask[i:i + mask_size, :] = 0

        if self.use_w:
            for j in range(0, w, grid_size):
                mask[:, j:j + mask_size] = 0

        return mask

    def _apply_edge_dilution(self, mask: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """
        Apply edge dilution: reduce masking near image borders (50px margin)
        """
        h, w = mask.shape
        edge_mask = np.ones_like(mask)

        # Mark edge regions
        edge_mask[:self.edge_margin, :] = 0  # Top
        edge_mask[-self.edge_margin:, :] = 0  # Bottom
        edge_mask[:, :self.edge_margin] = 0  # Left
        edge_mask[:, -self.edge_margin:] = 0  # Right

        # For boxes in edge regions, reduce masking
        for (x1, y1, x2, y2) in boxes:
            # Check if box intersects edge region
            in_edge = (x1 < self.edge_margin or x2 > w - self.edge_margin or
                       y1 < self.edge_margin or y2 > h - self.edge_margin)

            if in_edge:
                # Preserve this region (set mask to 1)
                mask[y1:y2, x1:x2] = 1

        return mask

    def _apply_mgm_to_image(
            self,
            img: np.ndarray,
            img_id: str,
            apply_edge_dilution: bool = True
    ) -> np.ndarray:
        """Apply MGM to a single image"""
        h, w = img.shape[:2]

        # Detect small objects
        small_boxes = self._detect_small_objects(img, img_id)

        # Create large mask (1/20 of shorter dimension)
        short_dim = min(h, w)
        large_grid_size = int(short_dim * self.large_mask_ratio)
        ratio = np.random.uniform(*self.ratio_range)
        large_mask = self._create_grid_mask(h, w, large_grid_size, ratio)

        # Apply edge dilution if enabled
        if apply_edge_dilution:
            large_mask = self._apply_edge_dilution(large_mask, small_boxes)

        # Create small masks for detected objects
        if small_boxes:
            # Find minimum box size
            min_box_size = min(
                min(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in small_boxes
            )
            small_grid_size = max(4, min_box_size // 2)  # Half of smallest box

            small_mask = self._create_grid_mask(h, w, small_grid_size, ratio)

            # Replace large mask with small mask in object regions
            for (x1, y1, x2, y2) in small_boxes:
                large_mask[y1:y2, x1:x2] = small_mask[y1:y2, x1:x2]

        # Apply mask to image
        if self.mode == 1:
            mask = large_mask
        else:
            mask = 1 - large_mask

        # Expand mask to 3 channels
        mask = mask[:, :, np.newaxis]

        # Apply mask
        img_masked = img * mask

        return img_masked.astype(img.dtype)

    def __call__(self, results: Dict) -> Dict:
        """
        Apply MGM to multi-view images in mmdet3d format

        Args:
            results: Dict containing 'img' key with shape [N, H, W, C]
                    where N is number of cameras

        Returns:
            Modified results dict
        """
        # Check if should apply augmentation
        prob = self._get_augmentation_probability()
        if np.random.rand() > prob:
            return results

        # Get images
        if 'img' not in results:
            return results

        imgs = results['img']  # [N, H, W, C] or list of images

        # Handle different formats
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:  # Single image [H, W, C]
                imgs = [imgs]
            elif imgs.ndim == 4:  # Multiple images [N, H, W, C]
                imgs = [imgs[i] for i in range(imgs.shape[0])]

        # Apply MGM to each image
        augmented_imgs = []
        for idx, img in enumerate(imgs):
            # Generate unique ID for caching
            img_id = f"{results.get('sample_idx', 0)}_{idx}"

            # Apply MGM
            img_aug = self._apply_mgm_to_image(img, img_id)
            augmented_imgs.append(img_aug)

        # Update results
        if isinstance(results['img'], np.ndarray):
            if results['img'].ndim == 4:
                results['img'] = np.stack(augmented_imgs, axis=0)
            else:
                results['img'] = augmented_imgs[0]
        else:
            results['img'] = augmented_imgs

        return results

    def set_epoch(self, epoch: int):
        """Update current epoch for probability scheduling"""
        self.current_epoch = epoch

        # Save cache periodically
        if epoch % 10 == 0:
            self._save_cache()

    def __repr__(self):
        return (f"MultiViewGridMask(prob_schedule={self.prob_schedule}, "
                f"edge_margin={self.edge_margin}, "
                f"ratio_range={self.ratio_range}, "
                f"density_range={self.density_range})")


# For mmdet3d pipeline registration
try:
    from mmdet.datasets.builder import PIPELINES

    PIPELINES.register_module()(MultiViewGridMask)
except ImportError:
    print("Warning: mmdet not installed, skipping pipeline registration")