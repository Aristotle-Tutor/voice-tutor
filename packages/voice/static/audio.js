// Connects this tab's microphone and speaker to the server over one websocket.
// The protocol is described in packages/voice/browser_audio.py.
window.voiceTutorAudio = (() => {
  const MIC_RATE = 16000;
  const SPEAKER_RATE = 24000;
  const WORKLETS = new URL("worklets.js", document.currentScript.src);
  let context = null;
  let socket = null;
  let stream = null;

  async function start() {
    if (context) return;
    try {
      await connect();
    } catch (error) {
      stop();
      throw error;
    }
  }

  async function connect() {
    context = new AudioContext();
    await context.audioWorklet.addModule(WORKLETS);
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const microphone = new AudioWorkletNode(context, "microphone", {
      numberOfOutputs: 0,
      processorOptions: { targetRate: MIC_RATE },
    });
    context.createMediaStreamSource(stream).connect(microphone);
    const speaker = new AudioWorkletNode(context, "speaker", {
      outputChannelCount: [1],
      processorOptions: { sourceRate: SPEAKER_RATE },
    });
    speaker.connect(context.destination);

    socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/audio`);
    socket.binaryType = "arraybuffer";
    const send = (data) => socket?.readyState === WebSocket.OPEN && socket.send(data);
    microphone.port.onmessage = (event) => send(event.data);
    speaker.port.onmessage = (event) => send(JSON.stringify(event.data));
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        speaker.port.postMessage(JSON.parse(event.data));
        return;
      }
      const utterance = new DataView(event.data).getUint32(0, true);
      const pcm = event.data.slice(4);
      speaker.port.postMessage({ type: "audio", utterance, pcm }, [pcm]);
    };
    socket.onclose = stop;
  }

  function stop() {
    socket?.close();
    stream?.getTracks().forEach((track) => track.stop());
    context?.close();
    socket = stream = context = null;
  }

  return { start, stop };
})();
