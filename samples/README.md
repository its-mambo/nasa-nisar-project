# Saved analysis sample

`judges/current.json` selects the immutable revision for playback. The app verifies the saved result and image checksums. Historical files remain unchanged; the live app and exported report display their detector evidence without the old model claims.

Run `python launch.py --mode sample` to replay the sample without weights or source HDF5 files. The sample uses native T0 ROI `[5323, 2414, 11255, 6066]` and threshold `0.10`. Source filenames and the fixed selection are defined in `nisar/features/samples/spec.py`.

The current bundle includes all ten PNGs (before, after, colour/grayscale,
composites, overlay, and cover preview), about 357 MiB in total. They are ordinary
Git files; Git LFS and an external download are not required. The server checks
all artifact hashes before offering playback. UTF-8 reads and byte-preserving
Git attributes keep this portable across operating systems.

## If the case study does not open

- **No dates or preview; Explore is disabled:** read the error below the button.
  In live mode, the app now shows why the saved bundle failed to load. Run
  `python launch.py --mode sample --no-chat --port 8001` from the same checkout
  to check it directly, then open [that server](http://127.0.0.1:8001).
- **Missing bundle or checksum mismatch:** use a complete, current clone/download
  of this repository. Keep `current.json` and its selected revision directory
  together. Older Windows checkouts may have rewritten JSON line endings; a
  fresh checkout with the included `.gitattributes` preserves the stored bytes.
  Do not regenerate the sample or edit its checksums to bypass validation.
- **App setup could not load:** serve the app through `launch.py`, rather than
  opening the HTML file directly or using a static file server. Reload after
  restarting. An occupied port can point your browser at an older server;
  `--port 8001` selects a separate one.
- **Dates appear but an image fails:** check the server terminal and run
  `python -m unittest tests.samples.test_playback -v`. This downloads every image
  from a temporary local server and verifies its checksum. Full-resolution
  images are roughly 41–48 MiB each, so a remote connection may take time.

## Regenerate as a maintainer

To generate a new revision, put the required GCOV products in `src/`, prepare `models/detector.safetensors`, and run `python cache_sample.py --refresh`. Use `--source-dir` for products stored elsewhere. To refresh only the report, run `python cache_sample.py --refresh-report`.
