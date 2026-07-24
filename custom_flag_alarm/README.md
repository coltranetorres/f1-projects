# F1 Break Reminder

A local VS Code extension: a green/yellow/red flag in the Secondary Side Bar
tracks how long you've been coding in the current window. Red means take a
break.

## One-time setup after installing

VS Code always adds a new extension's view to the primary Activity Bar first.
To move it to the Secondary Side Bar (upper-right area):

1. Open the Secondary Side Bar: `View` -> `Appearance` -> `Secondary Side Bar`.
2. Drag the "F1 Break Reminder" icon from the primary Activity Bar into the
   Secondary Side Bar.

## Settings

- `breakReminder.yellowThresholdMinutes` (default 60)
- `breakReminder.redThresholdMinutes` (default 90)

## Using your own sound files

Replace `media/green.wav`, `media/yellow.wav`, `media/red.wav` with any audio
files of your own (same filenames, `.wav` extension) and reload the window.

## Development

Run `node scripts/generate-tones.js` to regenerate the default synthesized
tones. Press F5 in VS Code to launch the Extension Development Host.
