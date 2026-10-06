"""Generate markdown report for dataset inspection."""

from pathlib import Path
import struct
from collections import Counter

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")

def get_png_info(filepath):
    with open(filepath, 'rb') as f:
        sig = f.read(8)
        if sig != b'\x89PNG\r\n\x1a\n':
            return None
        f.read(4) # len
        f.read(4) # IHDR
        w, h, bit_depth, color_type, comp, filter_m, interlace = struct.unpack('>IIBBBBB', f.read(13))
        color_map = {0: 'Grayscale', 2: 'RGB', 3: 'Palette (Indexed RGB)', 4: 'Grayscale+Alpha', 6: 'RGBA'}
        return {
            'width': w,
            'height': h,
            'bit_depth': bit_depth,
            'color_type': color_map.get(color_type, str(color_type)),
            'raw_color_type': color_type
        }

def generate_report():
    cxr_dir = DATA_ROOT / "CXR_png"
    masks_dir = DATA_ROOT / "masks"
    clin_dir = DATA_ROOT / "ClinicalReadings"
    test_dir = DATA_ROOT / "test"

    cxr_files = sorted([f for f in cxr_dir.iterdir() if f.is_file()])
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()])
    clin_files = sorted([f for f in clin_dir.iterdir() if f.is_file()])
    test_files = sorted([f for f in test_dir.iterdir() if f.is_file()])

    mcu_cxr = [f for f in cxr_files if f.name.startswith("MCUCXR")]
    chn_cxr = [f for f in cxr_files if f.name.startswith("CHNCXR")]

    mcu_table = []
    for f in mcu_cxr[:20]:
        info = get_png_info(f)
        mcu_table.append((f.name, f"{info['height']} x {info['width']}", info['color_type'], f"{info['bit_depth']}-bit"))

    chn_table = []
    for f in chn_cxr[:20]:
        info = get_png_info(f)
        chn_table.append((f.name, f"{info['height']} x {info['width']}", info['color_type'], f"{info['bit_depth']}-bit"))

    return mcu_table, chn_table

if __name__ == "__main__":
    mcu_table, chn_table = generate_report()
    print("MCU first 5:", mcu_table[:5])
    print("CHN first 5:", chn_table[:5])
