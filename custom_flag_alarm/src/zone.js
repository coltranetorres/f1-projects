function computeZone(elapsedMinutes, yellowThresholdMinutes, redThresholdMinutes) {
  if (elapsedMinutes >= redThresholdMinutes) {
    console.log(`Zone RED: ${elapsedMinutes} >= ${redThresholdMinutes}`);
    return 'red';
  }
  if (elapsedMinutes >= yellowThresholdMinutes) {
    console.log(`Zone YELLOW: ${elapsedMinutes} >= ${yellowThresholdMinutes}`);
    return 'yellow';
  }
  console.log(`Zone GREEN: ${elapsedMinutes} < ${yellowThresholdMinutes}`);
  return 'green';
}

module.exports = { computeZone };
