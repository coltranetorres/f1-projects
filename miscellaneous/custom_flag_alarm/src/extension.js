const vscode = require('vscode');
const { computeZone } = require('./zone');
const { FlagViewProvider } = require('./flagViewProvider');

const TICK_INTERVAL_MS = 10000;

function activate(context) {
  console.log('F1 Break Reminder activated');
  const provider = new FlagViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('f1BreakReminder.flagView', provider)
  );

  let startTime = Date.now();
  let testElapsedMinutes = null;

  provider.onReset(() => {
    startTime = Date.now();
    testElapsedMinutes = null;
  });

  provider.onSetTestTime((elapsedMinutes) => {
    testElapsedMinutes = elapsedMinutes;
  });

  const interval = setInterval(() => {
    const elapsedMinutes = testElapsedMinutes !== null ? testElapsedMinutes : (Date.now() - startTime) / 60000;
    const yellowThreshold = 50;
    const redThreshold = 60;
    console.log(`Elapsed: ${elapsedMinutes.toFixed(1)}m, Yellow: ${yellowThreshold}, Red: ${redThreshold}`);
    const color = computeZone(elapsedMinutes, yellowThreshold, redThreshold);
    provider.postTick(color, elapsedMinutes);
  }, TICK_INTERVAL_MS);

  context.subscriptions.push({ dispose: () => clearInterval(interval) });
}

function deactivate() {}

module.exports = { activate, deactivate };
