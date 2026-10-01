// Audio processors that run on the browser's audio thread.

// Resamples the microphone to 16-bit PCM at `targetRate` and posts it in 20 ms frames.
class Microphone extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.step = sampleRate / options.processorOptions.targetRate;
    this.position = 0;
    this.frame = new Int16Array(options.processorOptions.targetRate / 50);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    for (; this.position < input.length; this.position += this.step) {
      const i = Math.floor(this.position);
      const next = i + 1 < input.length ? input[i + 1] : input[i];
      const sample = input[i] + (next - input[i]) * (this.position - i);
      this.frame[this.filled++] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
      if (this.filled === this.frame.length) {
        const frame = this.frame.slice();
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.filled = 0;
      }
    }
    this.position -= input.length;
    return true;
  }
}

// Plays queued 16-bit PCM recorded at `sourceRate`, utterance by utterance, and
// reports how many samples of each utterance have actually been played.
class Speaker extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.step = options.processorOptions.sourceRate / sampleRate;
    this.queue = [];
    this.offset = 0;
    this.finished = new Map();
    this.stopped = new Set();
    this.sinceReport = 0;
    this.port.onmessage = (event) => this.receive(event.data);
  }

  played(utterance) {
    const current = this.queue[0]?.utterance === utterance ? Math.floor(this.offset) : 0;
    return (this.finished.get(utterance) ?? 0) + current;
  }

  report(type, utterance) {
    this.port.postMessage({ type, utterance, samples: this.played(utterance) });
  }

  receive(message) {
    if (message.type === "audio" && !this.stopped.has(message.utterance)) {
      const pcm = new Int16Array(message.pcm);
      this.queue.push({ utterance: message.utterance, samples: Float32Array.from(pcm, (s) => s / 0x8000) });
    } else if (message.type === "stop") {
      const played = this.played(message.utterance);
      this.stopped.add(message.utterance);
      if (this.queue[0]?.utterance === message.utterance) this.offset = 0;
      this.queue = this.queue.filter((chunk) => chunk.utterance !== message.utterance);
      this.port.postMessage({ type: "stopped", utterance: message.utterance, samples: played });
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) {
      const chunk = this.queue[0];
      if (!chunk) {
        output[i] = 0;
        continue;
      }
      output[i] = chunk.samples[Math.floor(this.offset)];
      this.offset += this.step;
      if (this.offset >= chunk.samples.length) {
        this.queue.shift();
        this.offset = 0;
        this.finished.set(chunk.utterance, (this.finished.get(chunk.utterance) ?? 0) + chunk.samples.length);
        if (!this.queue.some((next) => next.utterance === chunk.utterance)) this.report("played", chunk.utterance);
      }
    }
    this.sinceReport += output.length;
    if (this.queue[0] && this.sinceReport >= sampleRate / 10) {
      this.sinceReport = 0;
      this.report("played", this.queue[0].utterance);
    }
    return true;
  }
}

registerProcessor("microphone", Microphone);
registerProcessor("speaker", Speaker);
