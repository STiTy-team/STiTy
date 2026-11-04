/** [spoken language, what was said, its translation]. */
export type ScriptLine = readonly [language: 'ko' | 'en', original: string, translation: string];

/**
 * The app's introduction, played in every build: two people discovering what the screen shows,
 * using only real features.
 */
export const INTRO_SCRIPT: readonly ScriptLine[] = [
  [
    'en',
    'Hi! Does this app translate while we talk?',
    '안녕하세요! 이 앱이 대화하면서 바로 번역해 주나요?',
  ],
  [
    'ko',
    '네, 제가 한국어로 말하면 바로 영어로 보여 드려요.',
    'Yes. When I speak Korean, you see it in English right away.',
  ],
  ['en', 'So I can just keep speaking English?', '그럼 저는 그냥 영어로 계속 말하면 되나요?'],
  [
    'ko',
    '맞아요. 번역은 크게, 원래 말한 문장은 그 아래에 작게 나와요.',
    'Right. The translation is large, and what was actually said sits smaller below it.',
  ],
  ['en', 'What are the grey cards?', '회색 카드는 뭐예요?'],
  [
    'ko',
    '이 휴대폰 주인이 한 말이에요. 제대로 들렸는지 확인할 수 있게요.',
    "Those are what this phone's owner said, so they can check it was heard right.",
  ],
];

/** Sample situations for mocked builds; fixtures and the mocked live session replay them. */

export const SCRIPTS = {
  cafe: [
    [
      'en',
      'Hi, could I get an iced americano, please?',
      '안녕하세요, 아이스 아메리카노 한 잔 주시겠어요?',
    ],
    ['ko', '네, 사이즈는 어떤 걸로 드릴까요?', 'Sure, what size would you like?'],
    ['en', 'A large one. Do you have oat milk?', '큰 사이즈로요. 귀리 우유 있나요?'],
    ['ko', '네, 있어요. 오백 원 추가돼요.', 'Yes, we do. It is 500 won extra.'],
    ['en', 'That is fine. Can I pay by card?', '괜찮아요. 카드로 결제할 수 있나요?'],
    ['ko', '물론이죠. 여기 카드 꽂아 주세요.', 'Of course. Please insert your card here.'],
    ['en', 'How long will it take?', '얼마나 걸릴까요?'],
    ['ko', '5분 정도 걸려요. 진동벨 드릴게요.', 'About five minutes. Here is a buzzer for you.'],
  ],
  meeting: [
    [
      'ko',
      '지난주 출시 이후 사용자 피드백부터 볼게요.',
      "Let's start with user feedback since last week's launch.",
    ],
    [
      'en',
      'Most reports are about the sign-in screen on Android.',
      '대부분의 제보는 안드로이드 로그인 화면에 관한 거예요.',
    ],
    [
      'en',
      'I think the keyboard covers the code field.',
      '키보드가 인증 코드 입력란을 가리는 것 같아요.',
    ],
    ['ko', '그건 이번 주 안에 고칠 수 있어요.', 'We can fix that within this week.'],
    [
      'en',
      'Good. Then we move the records release to next Monday.',
      '좋아요. 그럼 기록 기능 출시는 다음 주 월요일로 옮기죠.',
    ],
    [
      'ko',
      '요약 기능은 서버 쪽 일정을 먼저 확인해야 해요.',
      'For summaries we need to check the server schedule first.',
    ],
    [
      'en',
      'I will ask the server team today and share it in the channel.',
      '오늘 서버 팀에 물어보고 채널에 공유할게요.',
    ],
    [
      'ko',
      '좋습니다. 다음 안건은 소음 환경 테스트예요.',
      'Great. The next item is testing in noisy places.',
    ],
  ],
  lecture: [
    [
      'ko',
      '오늘은 음성 인식이 어떻게 작동하는지 이야기하겠습니다.',
      "Today I'll talk about how speech recognition works.",
    ],
    [
      'ko',
      '먼저 소리는 짧은 구간으로 잘려서 숫자로 바뀝니다.',
      'First, the sound is cut into short frames and turned into numbers.',
    ],
    [
      'ko',
      '모델은 이 숫자들을 보고 다음 글자를 예측합니다.',
      'The model looks at these numbers and predicts the next characters.',
    ],
    [
      'ko',
      '스트리밍 인식은 문장이 끝나기 전에 결과를 보여 줍니다.',
      'Streaming recognition shows results before the sentence ends.',
    ],
    [
      'ko',
      '그래서 화면의 글자가 중간에 바뀌기도 합니다.',
      'That is why the text on screen sometimes changes midway.',
    ],
    [
      'ko',
      '문장이 확정되면 그때 번역이 붙습니다.',
      'Once a sentence is committed, the translation is attached.',
    ],
  ],
  market: [
    ['en', 'How much is this one?', '이거 얼마예요?'],
    ['ko', '만 오천 원이에요.', "It's fifteen thousand won."],
    ['en', 'Can you do twelve?', '만 이천 원에 되나요?'],
    ['ko', '두 개 사시면 그렇게 해 드릴게요.', "If you buy two, I'll do that."],
    ['en', 'Deal. Two, please.', '좋아요. 두 개 주세요.'],
  ],
  palace: [
    [
      'ko',
      '여기가 경복궁의 정문인 광화문입니다.',
      'This is Gwanghwamun, the main gate of Gyeongbokgung.',
    ],
    ['ko', '1395년에 처음 지어졌습니다.', 'It was first built in 1395.'],
    ['ko', '이제 근정전으로 이동하겠습니다.', 'Now we will move to Geunjeongjeon.'],
    [
      'ko',
      '왕이 신하들과 행사를 치르던 곳입니다.',
      'This is where the king held ceremonies with his officials.',
    ],
    ['ko', '사진은 자유롭게 찍으셔도 됩니다.', 'Feel free to take photos.'],
    [
      'ko',
      '다음은 경회루입니다. 연못 위의 누각이에요.',
      'Next is Gyeonghoeru, a pavilion over a pond.',
    ],
  ],
} satisfies Record<string, readonly ScriptLine[]>;
