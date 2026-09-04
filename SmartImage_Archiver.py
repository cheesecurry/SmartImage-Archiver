"""
SmartImage Archiver (Memory Optimized - RAM focus)
"""
import os
import shutil
import argparse
import zipfile
import tempfile
import logging
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import rarfile
import py7zr
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

# =================================================================
# 1. Constants
# =================================================================
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.avif'}
CONFIG_FILENAME = "config.json"

# Color definitions (ANSI escape sequences)
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# Custom exception for missing RAR tools
class RarToolMissingError(Exception):
    """Raised when unrar.exe is not found in the system PATH or working directory."""
    pass

# =================================================================
# 2. Utility Functions
# =================================================================

def detect_archive_format(file_path):
    file_path = Path(file_path)
    if zipfile.is_zipfile(file_path):
        return 'zip'
    try:
        if rarfile.is_rarfile(file_path):
            return 'rar'
    except Exception:
        pass
    try:
        if py7zr.is_7zfile(file_path):
            return '7z'
    except Exception:
        pass
    return None

# =================================================================
# 3. Worker Functions
# =================================================================
def calculate_ssim_score(orig_arr, converted_img):
    """Calculates SSIM score between two images (0-100)."""
    img2 = converted_img.convert("RGB")
    img2_arr = np.asarray(img2)

    if orig_arr.shape[:2] != img2_arr.shape[:2]:
        resized_img = img2.resize(converted_img.size)
        img2_arr = np.asarray(resized_img)

    score = ssim(orig_arr, img2_arr, channel_axis=-1)
    return score * 100

def get_best_quality_for_format_logic(orig_img, fmt_name, target_ssim, orig_size):
    """Finds the minimum quality using binary search with 'Last Resort' high-fidelity check."""
    orig_rgb = orig_img.convert("RGB")
    orig_arr = np.asarray(orig_rgb)
    ext_map = {'WEBP': 'webp', 'AVIF': 'avif'}
    ext = ext_map.get(fmt_name, 'webp')
    
    def get_stats(q, target_fmt, is_lossless=False, subsampling_val=None):
        buf = io.BytesIO()
        try:
            if target_fmt == 'WEBP':
                # lossless=True のときは quality 引数は無視されるが、明示的に渡す
                orig_img.save(buf, "WEBP", quality=q, lossless=is_lossless)
            else:
                params = {"quality": q}
                if subsampling_val:
                    params["subsampling"] = subsampling_val
                orig_img.save(buf, "AVIF", **params)
            buf.seek(0)
            with Image.open(buf) as comp_img:
                score = calculate_ssim_score(orig_arr, comp_img)
                size = buf.getbuffer().nbytes
                data = buf.getvalue()
            return score, size, data
        except Exception:
            return None

    # 1. Check Q=10 for early exit
    stats_10 = get_stats(10, fmt_name)
    if stats_10 and stats_10[0] >= target_ssim and stats_10[1] < orig_size:
        return {'path_data': stats_10[2], 'size': stats_10[1], 'score': stats_10[0], 'ext': ext, 'q_label': '10'}

    # 2. Check Q=100
    stats_100 = get_stats(100, fmt_name)
    if stats_100 is None:
        return None
    
    s100, z100, d100 = stats_100

    # 3. Standard Search (if Q=100 already meets target)
    if s100 >= target_ssim:
        low, high = 11, 99
        best_res = {'path_data': d100, 'size': z100, 'score': s100, 'ext': ext, 'q_label': '100'}
        
        while low <= high:
            mid = (low + high) // 2
            res = get_stats(mid, fmt_name)
            if res and res[0] >= target_ssim:
                best_res = {'path_data': res[2], 'size': res[1], 'score': res[0], 'ext': ext, 'q_label': str(mid)}
                high = mid - 1
            else:
                low = mid + 1
        return best_res

    # 4. "Last Resort" Check (Q=100 failed to hit target SSIM)
    last_resorts = []
    
    # Option A: WebP Lossless
    res_w_l = get_stats(100, 'WEBP', is_lossless=True)
    if res_w_l and res_w_l[0] >= target_ssim and res_w_l[1] < orig_size:
        last_resorts.append({'path_data': res_w_l[2], 'size': res_w_l[1], 'score': res_w_l[0], 'ext': 'webp', 'q_label': 'lossless'})
    
    # Option B: AVIF 4:4:4
    res_a_444 = get_stats(100, 'AVIF', subsampling_val="4:4:4")
    if res_a_444 and res_a_444[0] >= target_ssim and res_a_444[1] < orig_size:
        last_resorts.append({'path_data': res_a_444[2], 'size': res_a_444[1], 'score': res_a_444[0], 'ext': 'avif', 'q_label': '100 [4:4:4]'})

    if last_resorts:
        # 最もサイズが小さいものを選択
        best_last = min(last_resorts, key=lambda x: x['size'])
        return best_last

    # 5. Fallback: Q=100 (Standard)
    return {'path_data': d100, 'size': z100, 'score': s100, 'ext': ext, 'q_label': '100'}

def process_single_file_worker(file_path, rel_path, tmp_work_p, target_ssim, orig_size, requested_format=None):
    try:
        file_path = Path(file_path)
        rel_path = Path(rel_path)
        orig_img = Image.open(file_path)
        if orig_img.mode not in ("RGB", "L"):
            orig_img = orig_img.convert("RGB")
        
        candidates = []
        formats_to_test = [requested_format.upper()] if requested_format else ['WEBP', 'AVIF']

        for fmt in formats_to_test:
            res = get_best_quality_for_format_logic(orig_img, fmt, target_ssim, orig_size)
            if res and res['size'] < orig_size:
                candidates.append(res)

        if not candidates:
            dest = tmp_work_p / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            return ("SKIP", f"SKIP: {rel_path} (No better compression found)")

        best = min(candidates, key=lambda x: x['size'])
        new_rel_path = rel_path.with_suffix(f".{best['ext']}")
        final_dest = tmp_work_p / new_rel_path
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(final_dest, 'wb') as f:
            f.write(best['path_data'])
            
        reduction = ((orig_size - best['size']) / orig_size) * 100
        # Q=q_label を使用してログ出力
        return ("INFO", f"OK: {rel_path} -> {best['ext'].upper()} (Q={best['q_label']}, Size: {best['size']} bytes, SSIM={best['score']:.1f}, Reduction: {reduction:.2f}%)")

    except Exception as e:
        try:
            dest = Path(tmp_work_p) / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            return ("ERROR", f"ERR: {rel_path} ({str(e)}) - Copied original due to error")
        except Exception as copy_e:
            return ("ERROR", f"ERR: {rel_path} (Critical Error: {str(e)} | Copy Failed: {str(copy_e)})")

# =================================================================
# 4. Main Class
# =================================================================
class Converter:
    def __init__(self, target_ssim, log_file, requested_format=None, max_workers=None):
        self.target_ssim = target_ssim
        self.log_file = log_file
        self.requested_format = requested_format.upper() if requested_format else None
        self.max_workers = max_workers
        self.logger = logging.getLogger("SmartImageArchiver")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def process_archive(self, input_path, output_zip_path, show_progress=True):
        input_path = Path(input_path)
        output_zip_path = Path(output_zip_path)
        orig_archive_size = input_path.stat().st_size
        error_count = 0 
        
        with tempfile.TemporaryDirectory() as tmp_extract, \
             tempfile.TemporaryDirectory() as tmp_work:
            
            tmp_extract_p = Path(tmp_extract)
            tmp_work_p = Path(tmp_work)

            detected_fmt = detect_archive_format(input_path)
            if detected_fmt == 'zip':
                print(f"Extracting: {input_path.name}...")
                with zipfile.ZipFile(input_path, 'r') as zf:
                    zf.extractall(tmp_extract_p)
            elif detected_fmt == 'rar':
                print(f"Extracting: {input_path.name}...")
                try:
                    with rarfile.RarFile(input_path) as rf:
                        rf.extractall(tmp_extract_p)
                except rarfile.RarCannotExec:
                    raise RarToolMissingError("RAR extraction requires 'unrar.exe'.")
            elif detected_fmt == '7z':
                print(f"Extracting: {input_path.name}...")
                with py7zr.SevenZipFile(input_path, mode='r') as sz:
                    sz.extractall(path=tmp_extract_p)
            else:
                raise ValueError(f"Unsupported format: {detected_fmt}")

            all_files = [f for f in tmp_extract_p.rglob('*') if f.is_file()]
            
            print(f"Processing {len(all_files)} files (Max Workers: {self.max_workers or 'All'}, Target SSIM: {self.target_ssim:.1f}%)...")
            
            tmp_work_p_str = str(tmp_work_p)
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for f in all_files:
                    rel = f.relative_to(tmp_extract_p)
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        orig_size = f.stat().st_size
                        futures.append(executor.submit(
                            process_single_file_worker, 
                            str(f), str(rel), tmp_work_p_str, self.target_ssim, orig_size, self.requested_format
                        ))
                    else:
                        dest = tmp_work_p / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)

                pbar = tqdm(total=len(futures), desc="Converting") if show_progress else None

                for future in as_completed(futures):
                    res_level, res_msg = future.result()
                    if res_level == "INFO":
                        self.logger.info(res_msg)
                    elif res_level == "ERROR":
                        error_count += 1 
                        self.logger.error(res_msg)
                    else:
                        self.logger.info(res_msg)

                    if pbar:
                        pbar.update(1)
                if pbar: pbar.close()

            print(f"Archiving to: {output_zip_path.name}...")
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(tmp_work_p):
                    for f in files:
                        file_full_path = Path(root) / f
                        arcname = file_full_path.relative_to(tmp_work_p)
                        zf.write(file_full_path, arcname)

            final_archive_size = output_zip_path.stat().st_size
            total_reduction_pct = ((orig_archive_size - final_archive_size) / orig_archive_size) * 100
            
            error_display = f"Errors: {error_count}"
            if error_count > 0:
                error_display = f"{COLOR_RED}Errors: {error_count}{COLOR_RESET}"

            summary = (f"\n--- Summary ---\n"
                       f"Original: {orig_archive_size/1024/1024:.2f}MB | "
                       f"Output: {final_archive_size/1024/1024:.2f}MB | "
                       f"Reduction: {total_reduction_pct:.2f}% | "
                       f"{error_display}\n")
            
            self.logger.info(summary)
            print(summary)
            
            return error_count

def main():
    if os.name == 'nt':
        os.system('')

    parser = argparse.ArgumentParser(description="SmartImage Archiver: High-fidelity, SSIM-optimized image archiver")
    parser.add_argument("input", help="Input archive file")
    parser.add_argument("--ssim", type=float, help="SSIM threshold")
    parser.add_argument("--format", choices=['webp', 'avif'], help="Specify format")
    parser.add_argument("--workers", type=int, help="CPU cores")
    
    args = parser.parse_args()
    input_file = Path(args.input)

    if not input_file.exists():
        print(f"Error: File {input_file} not found.")
        return

    config_ssim, config_workers = 90.0, None

    if "__compiled__" in globals():
        base_dir = Path(__compiled__.containing_dir)
    else:
        base_dir = Path(__file__).resolve().parent

    config_path = base_dir / CONFIG_FILENAME

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                config_ssim = config_data.get("ssim", config_ssim)
                config_workers = config_data.get("workers", config_workers)
        except Exception as e:
            print(f"Warning: Failed to load {config_path.name} ({e}). Using defaults.")

    final_ssim = args.ssim if args.ssim is not None else config_ssim
    final_workers = args.workers if args.workers is not None else config_workers
    if final_workers is None:
        final_workers = max(1, (os.cpu_count() or 1) // 2)

    output_file = input_file.parent / (input_file.stem + "-convert.zip")
    log_file = input_file.parent / (input_file.stem + ".log")
    
    converter = Converter(final_ssim, log_file, args.format, final_workers)
    
    error_count = 0
    try:
        error_count = converter.process_archive(input_file, output_file)
        print(f"\nDone!")
    except RarToolMissingError as e:
        print(f"\n{COLOR_RED}Error: {str(e)}{COLOR_RESET}")
        print("Please download 'unrar.exe' and place it in the same folder as this program.")
        error_count = 1
    except Exception as e:
        print(f"\n{COLOR_RED}Error: {str(e)}{COLOR_RESET}")
        import traceback
        traceback.print_exc()
        error_count = 1 

    if error_count > 0:
        print(f"\nFinished with {error_count} error(s).")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
