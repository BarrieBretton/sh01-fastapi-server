# Catalog-inspired coverage

This document records what the AI Compose patch actually borrows from the
capability families represented by
[`awesome-claude-video-skills`](https://github.com/zhuyansen/awesome-claude-video-skills).
That repository is a catalog of independent projects, not a single runtime to
vendor. The catalog currently lists hundreds of skills across general
toolkits, promotion, explainers, editing, shorts, avatars, stories, motion
graphics, craft, and music.

## Implemented in this patch

- Local agent direction with a validated composition contract.
- Bounded draft/render/review/refinement iterations.
- Transcript- and silence-aware source selection.
- Versioned `sh01/ai-edit@2` program with source-to-program mappings.
- Word-timed captions and ASS export.
- Rendered hook, bridge, lower-third, quote, callout, and CTA graphics.
- Animated camera keyframe tracks rendered by FFmpeg as bounded zoom/pan
  expressions.
- Crossfade transitions for compatible base clips.
- Local and optional remote supplementary assets with provenance and partial
  frame geometry.
- Optional local SFX, Piper voiceover, and music input branches.
- Multi-variant output, evidence artifacts, receipts, feature counts, and
  mechanical render QC.
- A Kaggle notebook contact-sheet gate for human review of the actual MP4.

## Intentionally partial or future backends

These are represented in the notebook as explicit boundaries rather than
being advertised as complete:

- HTML/Remotion/HyperFrames rendering and browser/WebGL/Lottie scenes.
- A true multitrack editable project with undo/redo and FCPXML/project export.
- Beat-synchronous music editing and loudness/ducking automation.
- Independent visual-critic agents and an interactive storyboard approval UI.
- Reusable style-skill packs with a large template catalog.
- Avatar, lip-sync, generated-scene, and provider-specific video generation.

The same edit-program contract is designed so a future capability-selected
worker can consume it without adding another public caption or media endpoint.
The default worker remains safe, deterministic FFmpeg execution.
