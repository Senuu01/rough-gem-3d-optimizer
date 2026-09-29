"""Gemstone segmentation masks (Milestone 2 - NOT IMPLEMENTED YET).

Planned: GrabCut / background-subtraction masks saved in COLMAP's layout
(``masks/<image_name>.png``, black = ignored, white = used) and passed via
``--ImageReader.mask_path``. The command builder already supports the mask
path; TURNTABLE captures will not reconstruct reliably until masks exist,
because COLMAP will lock onto the static background.
"""
