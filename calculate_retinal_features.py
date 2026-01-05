#!/usr/bin/env python3
"""
Calculate ALL retinal vessel features from artery-vein segmentations.
Unified script for:
1. Disc Segmentation: Uses PVBM (if available) or fallback.
2. Feature Calculation: Uses standard methods for all features (CRAE, CRVE, Tortuosity, Fractal, Branching Angles).
   Does NOT use PVBM for feature calculation.

Usage:
    python calculate_all_retinal_features.py --base_dir outputs/disc_centered_av --output all_retinal_features.csv
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import argparse
import warnings
import traceback
from tqdm import tqdm

# PVBM Imports - ONLY for Disc Segmentation
try:
    from PVBM.DiscSegmenter import DiscSegmenter
    PVBM_AVAILABLE = True
except ImportError:
    PVBM_AVAILABLE = False
    print("⚠️  WARNING: PVBM library not found. Disc detection will use fallback method.")
    print("   Install via 'pip install PVBM'")

warnings.filterwarnings('ignore')

class UnifiedFeatureCalculator:
    def __init__(self, base_dir, optic_disc_diameter_um=1800, swap_av=False, use_pvbm=True):
        """
        Initialize feature calculator
        
        Args:
            base_dir: Base directory with segmentations (must contain artery_bin/ and vein_bin/)
            optic_disc_diameter_um: Optic disc diameter in micrometers (default 1800)
            swap_av: If True, swap artery and vein labels
            use_pvbm: If True, use PVBM for disc detection
        """
        self.base_dir = Path(base_dir)
        self.artery_bin_dir = self.base_dir / "artery_bin"
        self.vein_bin_dir = self.base_dir / "vein_bin"
        self.optic_disc_diameter_um = optic_disc_diameter_um
        self.swap_av = swap_av
        self.use_pvbm = use_pvbm and PVBM_AVAILABLE
        
        # Load summary.csv if available (for crop info/original image path)
        self.summary_path = self.base_dir / "summary.csv"
        if self.summary_path.exists():
            self.summary_df = pd.read_csv(self.summary_path)
        else:
            self.summary_df = None
            print("⚠️  summary.csv not found. Disc detection will rely on fallback methods.")

        # Initialize PVBM DiscSegmenter
        if self.use_pvbm:
            try:
                self.disc_segmenter = DiscSegmenter()
                print("✓ PVBM DiscSegmenter initialized")
            except Exception as e:
                print(f"⚠️  Error initializing PVBM DiscSegmenter: {e}")
                self.use_pvbm = False
                self.disc_segmenter = None

    def get_image_info(self, image_name):
        """Get original image path and crop parameters from summary.csv"""
        if self.summary_df is None:
            return None
            
        stem = Path(image_name).stem
        # Try exact match first
        image_rows = self.summary_df[self.summary_df['image'].astype(str).apply(lambda x: Path(x).stem == stem)]
        
        if len(image_rows) == 0:
            # Fallback to contains
            image_rows = self.summary_df[self.summary_df['image'].str.contains(stem)]
        
        if len(image_rows) == 0:
            return None
        
        row = image_rows.iloc[0]
        return {
            'orig_path': row['image'],
            'orig_width': int(row['orig_width']),
            'orig_height': int(row['orig_height']),
            'crop_x0': int(row['crop_x0']),
            'crop_y0': int(row['crop_y0']),
            'crop_side': int(row['crop_side'])
        }

    def load_segmentation(self, image_name):
        """Load artery and vein binary segmentations"""
        artery_path = self.artery_bin_dir / image_name
        vein_path = self.vein_bin_dir / image_name
        
        # Try png if jpg not found
        if not artery_path.exists():
            artery_path = self.artery_bin_dir / (Path(image_name).stem + ".png")
            vein_path = self.vein_bin_dir / (Path(image_name).stem + ".png")
            
        if not artery_path.exists() or not vein_path.exists():
            return None, None

        artery = cv2.imread(str(artery_path), cv2.IMREAD_GRAYSCALE)
        vein = cv2.imread(str(vein_path), cv2.IMREAD_GRAYSCALE)
        
        if artery is None or vein is None:
            return None, None
        
        # Binarize
        artery = (artery > 127).astype(np.uint8)
        vein = (vein > 127).astype(np.uint8)
        
        if self.swap_av:
            artery, vein = vein, artery
            
        return artery, vein

    # =========================================================================
    # OPTIC DISC DETECTION (PVBM or Fallback)
    # =========================================================================

    def detect_optic_disc_pvbm(self, image_name):
        """Detect optic disc using PVBM DiscSegmenter"""
        img_info = self.get_image_info(image_name)
        if not img_info:
            raise ValueError("No image info found in summary.csv")
            
        orig_image_path = img_info['orig_path']
        if not Path(orig_image_path).exists():
            raise FileNotFoundError(f"Original image not found: {orig_image_path}")
            
        orig_image = cv2.imread(orig_image_path)
        if orig_image is None:
            raise ValueError("Could not load original image")
            
        # Apply crop
        y0, x0, side = img_info['crop_y0'], img_info['crop_x0'], img_info['crop_side']
        cropped_image = orig_image[y0:y0+side, x0:x0+side]
        
        # Save temp for PVBM
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cropped_image)
            
        try:
            segmentation = self.disc_segmenter.segment(image_path=tmp_path)
            center, radius, _, _ = self.disc_segmenter.post_processing(segmentation, max_roi_size=600)
            Path(tmp_path).unlink()
            return int(center[1]), int(center[0]), float(radius) # y, x, r
        except Exception as e:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise e

    def detect_optic_disc_fallback(self, artery_mask, vein_mask):
        """Fallback disc detection using vessel convergence"""
        vessels = np.logical_or(artery_mask, vein_mask).astype(np.uint8) * 255
        h, w = vessels.shape
        
        # Gaussian blur + Hough Circle
        blurred = cv2.GaussianBlur(vessels, (21, 21), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h,w)//2,
                                  param1=50, param2=30, minRadius=int(min(h,w)*0.05), maxRadius=int(min(h,w)*0.15))
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            best_circle = None
            max_count = 0
            for (x, y, r) in circles:
                mask = np.zeros_like(vessels)
                cv2.circle(mask, (x, y), r, 255, -1)
                count = np.sum((vessels > 0) & (mask > 0))
                if count > max_count:
                    max_count = count
                    best_circle = (x, y, r)
            if best_circle is not None:
                return best_circle[1], best_circle[0], best_circle[2] # y, x, r
        
        # Center fallback
        return h//2, w//2, int(min(h,w)*0.075)

    # =========================================================================
    # FEATURE CALCULATION
    # =========================================================================

    def get_measurement_zone_mask(self, shape, cy, cx, r):
        h, w = shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Zone B: 0.5 to 1.0 disc diameters from margin (1.5r to 3.0r from center)
        return ((dist >= 2*r) & (dist <= 3*r)).astype(np.uint8)

    def measure_vessel_diameters(self, mask, um_per_pixel):
        if mask.sum() == 0: return []
        labeled = label(mask, connectivity=2)
        regions = regionprops(labeled)
        diameters = []
        for region in regions:
            if region.area < 20: continue
            comp_mask = (labeled == region.label).astype(np.uint8)
            dist = ndimage.distance_transform_edt(comp_mask)
            skel = skeletonize(comp_mask)
            vals = dist[skel > 0] * 2 * um_per_pixel
            if len(vals) > 0:
                d = np.median(vals)
                if d >= 25: diameters.append(d)
        return sorted(diameters, reverse=True)

    def calculate_knudtson(self, diameters):
        if not diameters: return 0
        vessels = diameters[:6]
        while len(vessels) > 1:
            new_vessels = []
            vessels = sorted(vessels, reverse=True)
            i, j = 0, len(vessels)-1
            while i < j:
                new_vessels.append(0.88 * np.sqrt(vessels[i]**2 + vessels[j]**2))
                i += 1; j -= 1
            if i == j: new_vessels.append(vessels[i])
            vessels = new_vessels
        return vessels[0]

    def calculate_tortuosity(self, binary_mask):
        """
        Calculate vessel tortuosity
        Tortuosity = (actual length) / (straight line distance)
        """
        if binary_mask.sum() == 0:
            return 0
        
        skeleton = skeletonize(binary_mask)
        labeled = label(skeleton, connectivity=2)
        regions = regionprops(labeled)
        
        tortuosities = []
        
        for region in regions:
            if region.area < 20: continue
            
            coords = np.argwhere(labeled == region.label)
            actual_length = region.area
            
            if len(coords) < 2: continue
            
            # Find endpoints
            component_mask = (labeled == region.label).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            neighbor_count = cv2.filter2D(component_mask, -1, kernel) - component_mask
            endpoints = np.argwhere((neighbor_count == 1) & (component_mask == 1))
            
            if len(endpoints) >= 2:
                straight_dist = np.linalg.norm(endpoints[0] - endpoints[-1])
                if straight_dist > 10:
                    tortuosity = actual_length / (straight_dist + 1e-6)
                    tortuosities.append(tortuosity)
        
        if len(tortuosities) == 0:
            return 1.0
        
        return np.median(tortuosities)

    def calculate_fractal_dimension(self, binary_mask):
        """
        Calculate fractal dimension using box-counting method
        """
        if binary_mask.sum() == 0: return 0
        
        skeleton = skeletonize(binary_mask)
        pixels = np.argwhere(skeleton > 0)
        
        if len(pixels) < 10: return 0
        
        scales = np.logspace(0.5, 3, num=15, base=2)
        counts = []
        
        for scale in scales:
            bins_x = np.arange(0, skeleton.shape[1], scale)
            bins_y = np.arange(0, skeleton.shape[0], scale)
            H, _, _ = np.histogram2d(pixels[:, 0], pixels[:, 1], bins=[bins_y, bins_x])
            counts.append(np.sum(H > 0))
            
        if len(counts) < 2: return 0
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        return -coeffs[0]

    def find_bifurcations(self, skeleton):
        """Find bifurcation points in skeleton"""
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(float), kernel, mode='constant')
        bifurcations = (neighbor_count >= 13) & (skeleton > 0)
        return np.argwhere(bifurcations)

    def calculate_branching_angles(self, binary_mask):
        """
        Calculate branching angles at bifurcation points
        Returns median angle in degrees
        """
        if binary_mask.sum() == 0: return 0
        
        skeleton = skeletonize(binary_mask)
        bifurcations = self.find_bifurcations(skeleton)
        
        if len(bifurcations) == 0: return 0
        
        angles = []
        for bif_point in bifurcations:
            y, x = bif_point
            window_size = 15
            y_min, y_max = max(0, y - window_size), min(skeleton.shape[0], y + window_size)
            x_min, x_max = max(0, x - window_size), min(skeleton.shape[1], x + window_size)
            
            local_skeleton = skeleton[y_min:y_max, x_min:x_max].copy()
            local_y, local_x = y - y_min, x - x_min
            local_skeleton[local_y, local_x] = 0
            
            labeled_branches = label(local_skeleton, connectivity=2)
            num_branches = labeled_branches.max()
            
            if num_branches < 2: continue
            
            branch_vectors = []
            for branch_id in range(1, min(num_branches + 1, 4)):
                branch_coords = np.argwhere(labeled_branches == branch_id)
                if len(branch_coords) < 2: continue
                
                distances = np.sqrt(((branch_coords - [local_y, local_x])**2).sum(axis=1))
                close_points = branch_coords[distances < 10]
                if len(close_points) < 2: continue
                
                direction = np.mean(close_points - [local_y, local_x], axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                branch_vectors.append(direction)
            
            if len(branch_vectors) >= 2:
                for i in range(len(branch_vectors) - 1):
                    v1, v2 = branch_vectors[i], branch_vectors[i + 1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angles.append(np.arccos(cos_angle) * 180 / np.pi)
        
        return np.median(angles) if angles else 0

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process_image(self, image_name):
        results = {'image': image_name}
        
        # Load masks
        artery, vein = self.load_segmentation(image_name)
        if artery is None:
            return None
            
        # 1. Detect Disc (PVBM or Fallback)
        try:
            if self.use_pvbm and self.disc_segmenter:
                cy, cx, r = self.detect_optic_disc_pvbm(image_name)
                results['detection_method'] = 'PVBM'
            else:
                cy, cx, r = self.detect_optic_disc_fallback(artery, vein)
                results['detection_method'] = 'Fallback'
        except:
            cy, cx, r = self.detect_optic_disc_fallback(artery, vein)
            results['detection_method'] = 'Fallback (Error)'
            
        results.update({'disc_center_y': cy, 'disc_center_x': cx, 'disc_radius': r})
        
        # 2. Calculate Features
        um_per_pixel = self.optic_disc_diameter_um / (2 * r)
        zone_mask = self.get_measurement_zone_mask(artery.shape, cy, cx, r)
        
        # Artery Features
        art_zone = artery * zone_mask
        art_diams = self.measure_vessel_diameters(art_zone, um_per_pixel)
        results['CRAE'] = self.calculate_knudtson(art_diams)
        results['Artery_Mean_Width'] = np.mean(art_diams[:6]) if art_diams else 0
        results['Artery_Fractal_Dim'] = self.calculate_fractal_dimension(artery)
        results['Artery_Tortuosity'] = self.calculate_tortuosity(artery)
        results['Artery_Branching_Angle'] = self.calculate_branching_angles(artery)
        
        # Vein Features
        vein_zone = vein * zone_mask
        vein_diams = self.measure_vessel_diameters(vein_zone, um_per_pixel)
        results['CRVE'] = self.calculate_knudtson(vein_diams)
        results['Vein_Mean_Width'] = np.mean(vein_diams[:6]) if vein_diams else 0
        results['Vein_Fractal_Dim'] = self.calculate_fractal_dimension(vein)
        results['Vein_Tortuosity'] = self.calculate_tortuosity(vein)
        results['Vein_Branching_Angle'] = self.calculate_branching_angles(vein)
        
        # AVR
        results['AVR'] = results['CRAE'] / results['CRVE'] if results['CRVE'] > 0 else 0
            
        return results

    def run(self, output_path):
        # Get list of images
        files = sorted(list(self.artery_bin_dir.glob("*.jpg")) + list(self.artery_bin_dir.glob("*.png")))
        print(f"Found {len(files)} images. Starting extraction...")
        
        all_results = []
        for f in tqdm(files):
            try:
                res = self.process_image(f.name)
                if res: all_results.append(res)
            except Exception as e:
                print(f"Error {f.name}: {e}")
                # traceback.print_exc()
                
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} rows to {output_path}")
        print(f"  Columns: {len(df.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Unified Retinal Features (Standard Methods + PVBM Disc Detection)")
    parser.add_argument('--base_dir', required=True, help="Directory containing artery_bin/ and vein_bin/")
    parser.add_argument('--output', required=True, help="Output CSV path")
    parser.add_argument('--no_pvbm', action='store_true', help="Disable PVBM disc detection (use fallback)")
    parser.add_argument('--swap_av', action='store_true', help="Swap Artery/Vein labels")
    
    args = parser.parse_args()
    
    calc = UnifiedFeatureCalculator(
        base_dir=args.base_dir,
        swap_av=args.swap_av,
        use_pvbm=not args.no_pvbm
    )
    calc.run(args.output)
