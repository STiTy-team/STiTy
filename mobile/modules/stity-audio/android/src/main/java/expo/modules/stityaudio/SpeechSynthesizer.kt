package expo.modules.stityaudio

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class SpeechSynthesizer(
  context: Context,
  private val onFinished: (String) -> Unit,
  private val onFailed: (String, String) -> Unit,
) {
  private val pending = mutableListOf<() -> Unit>()
  private var ready = false
  private var tts: TextToSpeech

  init {
    tts = create(context, GOOGLE_TTS_ENGINE)
  }

  fun speak(utteranceId: String, text: String, languageTag: String, rate: Float) {
    runWhenReady {
      val locale = Locale.forLanguageTag(languageTag)
      if (tts.isLanguageAvailable(locale) < TextToSpeech.LANG_AVAILABLE) {
        onFailed(utteranceId, "unsupported-language")
        return@runWhenReady
      }
      tts.language = locale
      tts.setSpeechRate(rate)
      tts.speak(text, TextToSpeech.QUEUE_ADD, null, utteranceId)
    }
  }

  fun stop() {
    synchronized(pending) { pending.clear() }
    if (ready) tts.stop()
  }

  fun shutdown() {
    stop()
    tts.shutdown()
  }

  // Samsung devices default to Samsung TTS, which silently falls back to another language when a
  // voice pack is missing. Google TTS is preferred and the system default used only without it.
  private fun create(context: Context, engine: String?): TextToSpeech {
    lateinit var instance: TextToSpeech
    instance = TextToSpeech(context, { status ->
      if (status == TextToSpeech.SUCCESS) {
        configure(instance)
      } else if (engine != null) {
        tts = create(context, null)
      }
    }, engine)
    return instance
  }

  private fun configure(instance: TextToSpeech) {
    // Voice-communication usage makes speech follow the communication device chosen by
    // AudioRouter (speaker or earpiece) instead of the media route.
    instance.setAudioAttributes(
      AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build(),
    )
    instance.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
      override fun onStart(utteranceId: String) = Unit

      override fun onDone(utteranceId: String) = onFinished(utteranceId)

      override fun onStop(utteranceId: String, interrupted: Boolean) =
        onFailed(utteranceId, "cancelled")

      @Deprecated("Deprecated in Java")
      override fun onError(utteranceId: String) = onFailed(utteranceId, "error")

      override fun onError(utteranceId: String, errorCode: Int) = onFailed(utteranceId, "error")
    })

    val queued = synchronized(pending) {
      ready = true
      pending.toList().also { pending.clear() }
    }
    queued.forEach { it() }
  }

  private fun runWhenReady(action: () -> Unit) {
    val runNow = synchronized(pending) {
      if (!ready) pending.add(action)
      ready
    }
    if (runNow) action()
  }

  private companion object {
    const val GOOGLE_TTS_ENGINE = "com.google.android.tts"
  }
}
