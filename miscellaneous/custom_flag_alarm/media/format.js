function formatElapsed(minutes) {
  const totalMinutes = Math.max(0, Math.floor(minutes));
  const hours = Math.floor(totalMinutes / 60);
  const mins = totalMinutes % 60;
  return hours + 'h ' + mins + 'm';
}

if (typeof module !== 'undefined') {
  module.exports = { formatElapsed };
}
