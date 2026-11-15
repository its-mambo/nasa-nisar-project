# Saved analysis sample

`judges/current.json` selects the immutable revision for playback. The app verifies the saved result and image checksums. Historical files remain unchanged; the live app and exported report display their detector evidence without the old model claims.

Run `python launch.py --mode sample` to replay the sample without weights or source HDF5 files. The sample uses native T0 ROI `[5323, 2414, 11255, 6066]` and threshold `0.10`. Source filenames and the fixed selection are defined in `nisar/features/samples/spec.py`.

To generate a new revision, put the required GCOV products in `src/`, prepare `models/detector.safetensors`, and run `python cache_sample.py --refresh`. Use `--source-dir` for products stored elsewhere. To refresh only the report, run `python cache_sample.py --refresh-report`.
