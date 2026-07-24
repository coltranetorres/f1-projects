(function () {
  const vscode = acquireVsCodeApi();
  const timerDisplay = document.getElementById('timer-display');
  const boxBoxAudio = document.getElementById('audio-box-box');

  console.log('Webview initialized');
  console.log('Timer display:', timerDisplay);
  console.log('Audio element:', boxBoxAudio);
  if (boxBoxAudio) {
    console.log('Audio src:', boxBoxAudio.src);
    // Test if audio can play
    setTimeout(() => {
      console.log('Testing audio playback...');
      boxBoxAudio.play()
        .then(() => console.log('Test audio play SUCCESS'))
        .catch((err) => console.error('Test audio play FAILED:', err));
    }, 1000);
  } else {
    console.error('CRITICAL: Audio element not found!');
  }

  let lastColor = 'green';
  let timerStartTime = null;
  let timerInterval = null;
  let initialElapsedMinutes = 0;

  function formatTimer(minutes) {
    const totalSeconds = Math.floor(minutes * 60);
    const hours = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    return `${String(hours).padStart(2, '0')} : ${String(mins).padStart(2, '0')} : ${String(secs).padStart(2, '0')}`;
  }

  function setColor(color) {
    const fullscreenMechanic = document.querySelector('.fullscreen-mechanic');
    fullscreenMechanic.classList.remove('fullscreen-mechanic-green', 'fullscreen-mechanic-yellow', 'fullscreen-mechanic-red');
    fullscreenMechanic.classList.add('fullscreen-mechanic-' + color);
  }

  function updateLiveTimer() {
    if (timerStartTime === null) return;
    const elapsedMs = Date.now() - timerStartTime;
    const elapsedMinutes = initialElapsedMinutes + (elapsedMs / 60000);
    timerDisplay.textContent = formatTimer(elapsedMinutes);
  }

  window.addEventListener('message', (event) => {
    const message = event.data;
    if (message.type === 'tick') {
      if (message.color !== lastColor) {
        console.log(`Color changed: ${lastColor} → ${message.color}`);
        setColor(message.color);
        if (boxBoxAudio) {
          console.log('Playing audio...');
          boxBoxAudio.currentTime = 0;
          boxBoxAudio.play()
            .then(() => console.log('Audio playing successfully'))
            .catch((err) => console.error('Audio play error:', err));
        } else {
          console.error('Audio element not found!');
        }
        lastColor = message.color;
      }
    }
  });

  document.getElementById('btn-start-time').addEventListener('click', () => {
    const hours = parseInt(document.getElementById('input-hours').value) || 0;
    const minutes = parseInt(document.getElementById('input-minutes').value) || 0;
    initialElapsedMinutes = hours * 60 + minutes;
    timerStartTime = Date.now();

    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(updateLiveTimer, 100);

    vscode.postMessage({ type: 'setTestTime', elapsedMinutes: initialElapsedMinutes });
  });

  document.getElementById('btn-reset-time').addEventListener('click', () => {
    document.getElementById('input-hours').value = '0';
    document.getElementById('input-minutes').value = '0';
    setColor('green');
    timerDisplay.textContent = formatTimer(0);
    timerStartTime = null;
    initialElapsedMinutes = 0;
    if (timerInterval) clearInterval(timerInterval);
    lastColor = 'green';
    vscode.postMessage({ type: 'reset' });
  });
})();
