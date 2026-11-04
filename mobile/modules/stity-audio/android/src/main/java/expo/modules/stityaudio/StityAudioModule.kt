package expo.modules.stityaudio

import android.Manifest
import android.content.Context
import expo.modules.interfaces.permissions.Permissions
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.types.Enumerable

enum class OutputRoute(val value: String) : Enumerable {
  SPEAKER("speaker"),
  EARPIECE("earpiece"),
}

class CaptureStartException(cause: Throwable) :
  CodedException("ERR_CAPTURE_START", cause.message, cause)

class StityAudioModule : Module() {
  private val context: Context
    get() = appContext.reactContext ?: throw Exceptions.ReactContextLost()

  private val router by lazy { AudioRouter(context) }

  private val capture = AudioCapture(
    onChunk = { data -> sendEvent(EVENT_AUDIO_CHUNK, mapOf("data" to data)) },
    onError = { message -> sendEvent(EVENT_CAPTURE_ERROR, mapOf("message" to message)) },
  )

  private var synthesizer: SpeechSynthesizer? = null

  private fun speech(): SpeechSynthesizer =
    synthesizer ?: SpeechSynthesizer(
      context,
      onFinished = { id -> sendEvent(EVENT_SPEECH_FINISHED, mapOf("utteranceId" to id)) },
      onFailed = { id, reason ->
        sendEvent(EVENT_SPEECH_FAILED, mapOf("utteranceId" to id, "reason" to reason))
      },
    ).also { synthesizer = it }

  override fun definition() = ModuleDefinition {
    Name("StityAudio")

    Events(EVENT_AUDIO_CHUNK, EVENT_CAPTURE_ERROR, EVENT_SPEECH_FINISHED, EVENT_SPEECH_FAILED)

    AsyncFunction("getMicrophonePermissionAsync") { promise: Promise ->
      Permissions.getPermissionsWithPermissionsManager(
        appContext.permissions,
        promise,
        Manifest.permission.RECORD_AUDIO,
      )
    }

    AsyncFunction("requestMicrophonePermissionAsync") { promise: Promise ->
      Permissions.askForPermissionsWithPermissionsManager(
        appContext.permissions,
        promise,
        Manifest.permission.RECORD_AUDIO,
      )
    }

    AsyncFunction("startCaptureAsync") { sampleRate: Int, chunkSamples: Int ->
      router.enterCommunicationMode(appContext.currentActivity)
      try {
        capture.start(sampleRate, chunkSamples)
      } catch (e: Exception) {
        throw CaptureStartException(e)
      }
    }

    Function("stopCapture") {
      capture.stop()
    }

    Function("setOutputRoute") { route: OutputRoute ->
      when (route) {
        OutputRoute.SPEAKER -> {
          router.routeToSpeaker()
          capture.setPreferredInput(null)
        }
        OutputRoute.EARPIECE -> {
          router.routeToEarpiece()
          capture.setPreferredInput(router.builtInMicrophone())
        }
      }
    }

    Function("isHeadsetConnected") {
      router.isHeadsetConnected()
    }

    Function("releaseAudioSession") {
      router.release(appContext.currentActivity)
    }

    Function("speak") { utteranceId: String, text: String, languageTag: String, rate: Float ->
      speech().speak(utteranceId, text, languageTag, rate)
    }

    Function("stopSpeaking") {
      synthesizer?.stop()
    }

    OnDestroy {
      capture.stop()
      synthesizer?.shutdown()
      synthesizer = null
    }
  }

  private companion object {
    const val EVENT_AUDIO_CHUNK = "onAudioChunk"
    const val EVENT_CAPTURE_ERROR = "onCaptureError"
    const val EVENT_SPEECH_FINISHED = "onSpeechFinished"
    const val EVENT_SPEECH_FAILED = "onSpeechFailed"
  }
}
