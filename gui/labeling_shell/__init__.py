"""labeling_shell — the reusable, pipeline-agnostic desktop labeling shell.

This is the Python/PySide6 successor to the Electron `behavior-labeler` shell
(`LabelingApp/core/*.js`). Like the JS shell it is deliberately free of any
model/pipeline specifics: video playback, bouts/behaviors, timeline, project
I/O and CSV live here; a per-app pipeline plugin registers against it.

Phase 0 populates only the video engine (`video_reader`). Later phases add the
Qt widgets, bout model, and plugin seam.
"""
