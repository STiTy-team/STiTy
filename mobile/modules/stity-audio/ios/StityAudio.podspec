Pod::Spec.new do |s|
  s.name           = 'StityAudio'
  s.version        = '1.0.0'
  s.summary        = 'Microphone capture, audio routing and speech synthesis for STiTy'
  s.description    = 'Streams 16 kHz mono PCM from the microphone, routes output between speaker and earpiece, and speaks translations through the shared voice-chat audio session.'
  s.author         = 'STiTy'
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.license        = 'MIT'
  s.platforms      = { :ios => '16.4' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  s.source_files = "**/*.{h,m,mm,swift}"
end
