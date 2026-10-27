package expo.modules.stityaudio

import android.app.Activity
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build

class AudioRouter(context: Context) {
  private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

  // The session stays in MODE_IN_COMMUNICATION across pause and resume. Cycling the mode on
  // every pause made some Samsung devices lose the route, so only release() returns to normal.
  fun enterCommunicationMode(activity: Activity?) {
    if (audioManager.mode != AudioManager.MODE_IN_COMMUNICATION) {
      audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
      routeToSpeaker()
    }
    activity?.runOnUiThread { activity.volumeControlStream = AudioManager.STREAM_VOICE_CALL }
  }

  fun release(activity: Activity?) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      audioManager.clearCommunicationDevice()
    } else {
      @Suppress("DEPRECATION")
      audioManager.isSpeakerphoneOn = false
    }
    audioManager.mode = AudioManager.MODE_NORMAL
    activity?.runOnUiThread {
      activity.volumeControlStream = AudioManager.USE_DEFAULT_STREAM_TYPE
    }
  }

  fun routeToSpeaker() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      audioManager.availableCommunicationDevices
        .firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
        ?.let { audioManager.setCommunicationDevice(it) }
    } else {
      @Suppress("DEPRECATION")
      audioManager.isSpeakerphoneOn = true
    }
  }

  fun routeToEarpiece() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      val headset = audioManager.availableCommunicationDevices
        .firstOrNull { it.type in COMMUNICATION_HEADSET_TYPES }
      if (headset != null) {
        audioManager.setCommunicationDevice(headset)
      } else {
        audioManager.clearCommunicationDevice()
      }
    } else {
      @Suppress("DEPRECATION")
      audioManager.isSpeakerphoneOn = false
    }
  }

  // With output on a Bluetooth headset the platform would also move capture to the headset
  // microphone. Pinning the built-in microphone keeps both speakers on the phone's mic.
  fun builtInMicrophone(): AudioDeviceInfo? =
    audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
      .firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }

  fun isHeadsetConnected(): Boolean =
    audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
      .any { it.type in OUTPUT_HEADSET_TYPES }

  private companion object {
    val COMMUNICATION_HEADSET_TYPES = setOf(
      AudioDeviceInfo.TYPE_WIRED_HEADSET,
      AudioDeviceInfo.TYPE_WIRED_HEADPHONES,
      AudioDeviceInfo.TYPE_USB_HEADSET,
      AudioDeviceInfo.TYPE_BLUETOOTH_SCO,
      AudioDeviceInfo.TYPE_BLE_HEADSET,
    )
    val OUTPUT_HEADSET_TYPES = setOf(
      AudioDeviceInfo.TYPE_WIRED_HEADSET,
      AudioDeviceInfo.TYPE_WIRED_HEADPHONES,
      AudioDeviceInfo.TYPE_USB_HEADSET,
      AudioDeviceInfo.TYPE_BLUETOOTH_A2DP,
      AudioDeviceInfo.TYPE_BLUETOOTH_SCO,
      AudioDeviceInfo.TYPE_BLE_HEADSET,
    )
  }
}
