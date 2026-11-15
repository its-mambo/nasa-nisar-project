#!/usr/bin/env python3
"""Write a detector-only safetensors model from a packed NISAR checkpoint."""
from nisar.infrastructure.models.detector_weights import main

if __name__ == "__main__":
    main()
