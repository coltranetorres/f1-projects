const fs = require('fs');
const path = require('path');

const SAMPLE_RATE = 44100;

function envelope(i, n, attack, release) {
  const attackSamples = Math.floor(attack * SAMPLE_RATE);
  const releaseSamples = Math.floor(release * SAMPLE_RATE);
  if (i < attackSamples) return i / attackSamples;
  if (i > n - releaseSamples) return Math.max(0, (n - i) / releaseSamples);
  return 1;
}

function tone(freq, durationSeconds, attack, release, amplitude) {
  const n = Math.floor(durationSeconds * SAMPLE_RATE);
  const samples = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    samples[i] = Math.sin(2 * Math.PI * freq * t) * amplitude * envelope(i, n, attack, release);
  }
  return samples;
}

function silence(durationSeconds) {
  const n = Math.floor(durationSeconds * SAMPLE_RATE);
  return new Array(n).fill(0);
}

function concat(...chunks) {
  return chunks.reduce((acc, c) => acc.concat(c), []);
}

function writeWav(filePath, samples) {
  const dataSize = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(SAMPLE_RATE, 24);
  buffer.writeUInt32LE(SAMPLE_RATE * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(clamped * 32767), 44 + i * 2);
  }
  fs.writeFileSync(filePath, buffer);
}

const mediaDir = path.join(__dirname, '..', 'media');

// Green: single soft chime
writeWav(path.join(mediaDir, 'green.wav'), tone(880, 0.3, 0.02, 0.2, 0.4));

// Yellow: two short beeps
writeWav(
  path.join(mediaDir, 'yellow.wav'),
  concat(tone(660, 0.15, 0.01, 0.05, 0.5), silence(0.08), tone(660, 0.15, 0.01, 0.05, 0.5))
);

// Red: urgent alternating alarm, repeated 3 times
writeWav(
  path.join(mediaDir, 'red.wav'),
  concat(
    tone(880, 0.15, 0.005, 0.03, 0.6),
    tone(440, 0.15, 0.005, 0.03, 0.6),
    tone(880, 0.15, 0.005, 0.03, 0.6),
    tone(440, 0.15, 0.005, 0.03, 0.6),
    tone(880, 0.15, 0.005, 0.03, 0.6),
    tone(440, 0.15, 0.005, 0.03, 0.6)
  )
);

console.log('Generated green.wav, yellow.wav, red.wav in', mediaDir);
