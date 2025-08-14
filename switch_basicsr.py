#!/usr/bin/env python3
import argparse
from pathlib import Path

"""
Usage:
  python switch_basicsr.py --mode {sd, rf}
    
    optional: : 
        --nafnet_root ./NAFNet
            
"""

def detect_mode(nafnet_root: Path) -> str:
    """
    Detect current mode based on existing dirs.
    Returns: "sd", "rf", or "unknown"
    """
    bs      = nafnet_root / "basicsr"
    bs_sd   = nafnet_root / "basicsr_SD"
    bs_rf   = nafnet_root / "basicsr_RF"

    if bs.exists() and bs_rf.exists():
        return "sd"   # basicsr is SD, RF folder exists as alternative
    elif bs.exists() and bs_sd.exists():
        return "rf"   # basicsr is RF, SD folder exists as alternative
    else:
        return "unknown"
    
def switch_to(root: Path, target: str):
    bs      = root / "basicsr"
    bs_sd   = root / "basicsr_SD"
    bs_rf   = root / "basicsr_RF"

    current = detect_mode(root)

    if target == "sd":
        if current == "sd":
            print("Already in SD mode. No changes made.")
            return
        if not bs_sd.exists():
            raise FileNotFoundError(f"{bs_sd} not found. Cannot switch to SD.")
        print("Switching to SD mode (for Standalone deblurring)...")
        bs.rename(bs_rf)   # current basicsr -> basicsr_RF
        bs_sd.rename(bs)   # basicsr_SD -> basicsr
        print("Switched to SD mode.")

    elif target == "rf":
        if current == "rf":
            print("Already in RF mode. No changes made.")
            return
        if not bs_rf.exists():
            raise FileNotFoundError(f"{bs_rf} not found. Cannot switch to RF.")
        print("Switching to RF mode (for RF-guided deblurring)...")
        bs.rename(bs_sd)   # current basicsr -> basicsr_SD
        bs_rf.rename(bs)   # basicsr_RF -> basicsr
        print("Switched to RF mode.")

    else:
        raise ValueError("Target must be 'sd' or 'rf'.")

def main():
    ap = argparse.ArgumentParser(description="Fast rename-based switch between SD and RF basicsr.")
    ap.add_argument("--nafnet_root", default="./NAFNet", type=str, help="Path to NAFNet repo root.")
    ap.add_argument("--mode", required=True, choices=["sd", "rf"], help="Target mode.")
    args = ap.parse_args()

    root = Path(args.nafnet_root).resolve()
    switch_to(root, args.mode)

if __name__ == "__main__":
    main()
