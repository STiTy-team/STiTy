package expo.modules.stityaudio

import android.annotation.SuppressLint
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.util.Base64

class AudioCapture(
  private val onChunk: (String) -> Unit,
  private val onError: (String) -> Unit,
) {
  private var recorder: AudioRecord? = null
  private var echoCanceler: AcousticEchoCanceler? = null
  private var thread: Thread? = null

  @Volatile
  private var running = false

  val isRunning: Boolean
    get() = running

  @SuppressLint("MissingPermission")
  @Synchronized
  fun start(sampleRate: Int, chunkSamples: Int) {
    if (running) return

    val chunkBytes = chunkSamples * BYTES_PER_SAMPLE
    val minBuffer = AudioRecord.getMinBufferSize(sampleRate, CHANNEL, ENCODING)
    if (minBuffer <= 0) {
      throw IllegalStateException("Sample rate $sampleRate Hz is not supported by this device")
    }

    // VOICE_COMMUNICATION shares the microphone with call apps such as Zoom, which hold
    // VOICE_RECOGNITION exclusively, and gives the platform echo canceller a reference signal.
    val record = AudioRecord(
      MediaRecorder.AudioSource.VOICE_COMMUNICATION,
      sampleRate,
      CHANNEL,
      ENCODING,
      maxOf(minBuffer, chunkBytes) * 3,
    )
    if (record.state != AudioRecord.STATE_INITIALIZED) {
      record.release()
      throw IllegalStateException("AudioRecord failed to initialize")
    }

    if (AcousticEchoCanceler.isAvailable()) {
      echoCanceler = AcousticEchoCanceler.create(record.audioSessionId)?.apply { enabled = true }
    }

    recorder = record
    running = true
    record.startRecording()

    thread = Thread({ readLoop(record, chunkBytes) }, "StityAudioCapture").apply { start() }
  }

  @Synchronized
  fun stop() {
    running = false
    thread?.join(JOIN_TIMEOUT_MS)
    thread = null
    echoCanceler?.release()
    echoCanceler = null
    recorder?.run {
      if (recordingState == AudioRecord.RECORDSTATE_RECORDING) stop()
      release()
    }
    recorder = null
  }

  @Synchronized
  fun setPreferredInput(device: AudioDeviceInfo?) {
    recorder?.preferredDevice = device
  }

  private fun readLoop(record: AudioRecord, chunkBytes: Int) {
    val buffer = ByteArray(chunkBytes)
    while (running) {
      val read = record.read(buffer, 0, buffer.size)
      when {
        read > 0 -> onChunk(Base64.encodeToString(buffer, 0, read, Base64.NO_WRAP))
        read < 0 -> {
          running = false
          onError("AudioRecord.read failed with code $read")
        }
      }
    }
  }

  private companion object {
    const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
    const val BYTES_PER_SAMPLE = 2
    const val JOIN_TIMEOUT_MS = 500L
  }
}
