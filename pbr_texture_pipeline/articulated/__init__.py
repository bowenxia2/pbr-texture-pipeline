"""Articulated (PartNet-Mobility URDF) support: parsing, grouping, FK, stages.

Everything here is CUDA-free at import time (repo convention); heavy imports stay inside
functions. See PRD.md (articulated extension) for the design.
"""
