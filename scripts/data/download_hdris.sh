#!/bin/bash
# Download additional studio HDRIs from Poly Haven (CC0 license)
# These are 1k resolution for faster loading

HDRI_DIR="/Users/max/Code/ChessR/hdri"
mkdir -p "$HDRI_DIR"
cd "$HDRI_DIR"

echo "Downloading studio HDRIs from Poly Haven..."

# Studio HDRIs (good for indoor chess scenes)
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/industrial_sunset_puresky_1k.hdr" -o "industrial_sunset_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_03_1k.hdr" -o "studio_small_03_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_04_1k.hdr" -o "studio_small_04_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_05_1k.hdr" -o "studio_small_05_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_06_1k.hdr" -o "studio_small_06_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/photography_studio_01_1k.hdr" -o "photography_studio_01_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/modern_buildings_2_1k.hdr" -o "modern_buildings_2_1k.hdr"
curl -L "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/empty_warehouse_01_1k.hdr" -o "empty_warehouse_01_1k.hdr"

echo ""
echo "Download complete!"
echo "Total HDRIs in $HDRI_DIR:"
ls -1 *.hdr | wc -l
echo ""
echo "HDRI files:"
ls -lh *.hdr
