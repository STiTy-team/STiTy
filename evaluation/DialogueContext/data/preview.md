# DialogueContext — data preview

`scripts/author_dialogues.py` 가 만든다. 직접 고치지 않는다.

대화 5편, 턴 88개, 인스턴스 20개.

## 태그별 인스턴스 수

| 태그 | 인스턴스 |
|---|---|
| `pronoun_coreference` | 4 — c01-t06, c01-t14, c02-t12, c05-t14 |
| `omitted_argument` | 5 — c01-t08, c02-t10, c03-t05, c03-t14, c03-t17 |
| `gender_reference` | 5 — c01-t06, c01-t14, c04-t08, c04-t13, c05-t12 |
| `register_politeness` | 3 — c02-t12, c03-t14, c05-t14 |
| `lexical_consistency` | 3 — c02-t07, c03-t12, c05-t13 |
| `word_sense` | 3 — c03-t05, c03-t12, c04-t05 |
| `fragment_incremental` | 3 — c02-t10, c03-t17, c05-t12 |
| `discourse_connective` | 3 — c01-t12, c02-t15, c04-t12 |
| `entity_consistency` | 3 — c02-t07, c05-t09, c05-t13 |
| `context_trap` | 4 — c01-t14, c02-t15, c04-t13, c05-t09 |

## c01 — 친구 사이 수다 — 여자친구 생일 선물을 두고 나온 이야기 (반말)

- **A**: 준호 (남, 20대 후반, 회사원)
- **B**: 다은 (여, 20대 후반, 준호의 대학 친구)

| # | 화자 | ko | en | 조각 | 목표 |
|---|---|---|---|---|---|
| 0 | A | 야, 나 어제 진짜 정신없었어. | Man, yesterday was total chaos. |  |  |
| 1 | B | 왜, 무슨 일 있었어? | Why, what happened? |  |  |
| 2 | A | 수진이 생일이라 저녁 예약해 놨었거든. | It was Sujin's birthday, so I'd booked us a dinner. |  |  |
| 3 | A | 근데 식당 가는 길에 딱 생각난 거야, | But on the way to the restaurant, it suddenly hit me, | → |  |
| 4 | A | 여자친구 주려고 산 양말을 집에 두고 왔다는 게. | I'd left the socks I bought for my girlfriend at home. |  |  |
| 5 | B | 양말? 선물로 양말 산 거야? | Socks? You got socks as a present? |  |  |
| 6 | A | 그냥 그런 거 아니고, 걔가 옛날부터 갖고 싶어 하던 거야. | Not just any socks. They're the ones she's wanted forever. |  | **c01-t06** |
| 7 | B | 그래서 어떻게 했어? | So what'd you do? |  |  |
| 8 | A | 당연히 다시 가지러 갔지. | I went back for them, obviously. |  | **c01-t08** |
| 9 | A | 근데 그러고 나니까 시간이 너무 빠듯해서, | But after that I was cutting it really close, so | → |  |
| 10 | A | 결국 택시를 탔어. | I ended up taking a taxi. |  |  |
| 11 | B | 택시 탔으면 여자친구 안 기다리게 했겠네. | Well, at least with a taxi you didn't keep your girlfriend waiting. |  |  |
| 12 | A | 그렇기는 한데 결국 늦었어. | You'd think so, but I was still late. |  | **c01-t12** |
| 13 | A | 기사님이 골목길로 막 돌아가 주셨는데도 길이 너무 막혔어. | The driver kept cutting through back alleys and everything, but traffic was just awful. |  |  |
| 14 | B | 아이고. 그래서 걔는 좋아했어? | Oh no. So did she like them? |  | **c01-t14** |
| 15 | A | 응, 바로 신어 보더니 딱 맞는다고 엄청 좋아하더라. | Yeah, she put them on right away, said they fit perfectly, and totally loved them. |  |  |
| 16 | B | 다행이다. 다음엔 전날 미리 챙겨 놔. | Phew. Next time, get them ready the night before. |  |  |

### c01-t06 — `gender_reference`, `pronoun_coreference`

- 현재 발화 (A): 그냥 그런 거 아니고, 걔가 옛날부터 갖고 싶어 하던 거야.
- 정답: Not just any socks. They're the ones she's wanted forever.
- check `gender_reference`: Refers to 걔 (the girlfriend, Sujin) as 'she/her'; fails with 'he', 'they' or 'it'.
- check `pronoun_coreference`: 거 is resolved to the socks: plural 'they/the ones/socks' or 'a pair', not a singular 'it', 'something' or 'the thing' standing for them.
- 문맥 효과: n=0 has neither referent: typically 'it's something he's wanted for a long time'. n=1 (B: 'Socks?') fixes the object but not the gender. n=3 reaches turn 4 ('my girlfriend') and should fix both.
- 메모: Gender cue '여자친구' is 2 turns back (turn 4); 'socks' is 1 back. The current turn deliberately avoids the word 양말.

### c01-t08 — `omitted_argument`

- 현재 발화 (A): 당연히 다시 가지러 갔지.
- 정답: I went back for them, obviously.
- check `omitted_argument`: The omitted object is the socks: 'them' or 'the socks'; fails with 'it', 'something' or no object that makes sense.
- 문맥 효과: n=0 gives 'I went back to get it.' The socks are 2-3 turns back (turns 5-6), so n=1 ('So what'd you do?') still misses; n=3 should fix.
- 메모: STiTy-style example adapted from '그래서 다시 가지러 갔지.' Object cue distance 2-3 (turns 5-6).

### c01-t12 — `discourse_connective`

- 현재 발화 (A): 그렇기는 한데 결국 늦었어.
- 정답: You'd think so, but I was still late.
- check `discourse_connective`: 그렇기는 한데 is rendered as conceding B's assumption while contradicting it ('You'd think so, but…', 'It should have, but…', 'That was the idea, but…'); a hedged 'Yeah, but I was still late' is also fine. Fails only with an explicit affirmation of B's claim ('That's true, but…', 'You're right, but…') that then contradicts itself.
- 문맥 효과: n=0 cannot know what is being conceded and usually outputs 'That's true, but I ended up being late.' n=1 (B: 'at least with a taxi you didn't keep your girlfriend waiting') is enough to pick the counter-expectation reading.
- 메모: STiTy-style example adapted from prev '그래서 택시를 탔어.' → '그렇기는 한데 결국 늦었어.' Cue distance 1 (turn 11), taxi at 2.

### c01-t14 — `gender_reference`, `pronoun_coreference`, `context_trap`

- 현재 발화 (B): 아이고. 그래서 걔는 좋아했어?
- 정답: Oh no. So did she like them?
- check `gender_reference`: Refers to 걔 as female ('she/her', 'your girlfriend' or 'Sujin'); fails with 'he' or 'they'.
- check `pronoun_coreference`: 걔 refers to the girlfriend (e.g., 'did she like them', 'was your girlfriend happy'), not to the taxi driver.
- check `context_trap`: Does NOT pull in the driver or the lateness (no 'even though you were late', no mention of the driver or traffic).
- 문맥 효과: n=0 defaults to 'did he like it?'. n=1 shows only the driver (turn 13), the most recent person, which invites 'he' = the driver. n=3 reaches turn 11 ('your girlfriend') and should fix the gender and referent.
- 메모: Distractor: the taxi driver is the most recently mentioned person. Gender cue '여자친구' is 3 turns back (turn 11).

## c02 — 직장 — 팀장(반말)과 사원(존댓말), 거래처 배너 시안과 워크숍 이야기

- **A**: 박준영 팀장 (남, 40대, 마케팅팀 팀장)
- **B**: 이서윤 (여, 20대 후반, 마케팅팀 사원)

| # | 화자 | ko | en | 조각 | 목표 |
|---|---|---|---|---|---|
| 0 | A | 서윤 씨, 잠깐 시간 돼? | Seoyun, got a sec? |  |  |
| 1 | B | 네, 팀장님. 무슨 일이세요? | Yes, of course. What is it? |  |  |
| 2 | A | 우리가 초록병이라고 부르는 음료 회사 있잖아. | You know that beverage company we call Green Bottle? |  |  |
| 3 | A | 거기서 배너 시안이 다 마음에 안 든대. | They say they don't like any of the banner mock-ups. |  |  |
| 4 | B | 네? 어제 시안 세 개 다 보내 드렸는데요. | Sorry? I sent over all three mock-ups yesterday. |  |  |
| 5 | A | 응, 셋 다 색이 너무 튄다고. | Yeah, they said the colors on all three are too loud. |  |  |
| 6 | B | 아... 그럼 톤을 좀 차분하게 다시 잡아 볼게요. | Oh... Then I'll rework them in a calmer palette. |  |  |
| 7 | B | 초록병 수정 시안은 오후에 올려 드리면 될까요? | Would it be okay if I got you the revised Green Bottle mock-ups this afternoon? |  | **c02-t07** |
| 8 | A | 그래, 네 시 전에만 줘. | Sure, just get them to me before four. |  |  |
| 9 | A | 아, 그리고 내일 워크숍 가는 버스, | Oh, and the bus for tomorrow's workshop | → |  |
| 10 | A | 여덟 시에 회사 앞에서 출발이야. | leaves from in front of the office at eight. |  | **c02-t10** |
| 11 | B | 네, 팀장님. 공지 봤습니다. | Yes, I saw the announcement. |  |  |
| 12 | B | 근데 팀장님도 같이 가세요? | Are you coming with us too? |  | **c02-t12** |
| 13 | A | 나? 나는 오후에 합류해. 오전에 본부장님 보고가 있거든. | Me? I'll join in the afternoon. I've got a briefing with the division head in the morning. |  |  |
| 14 | B | 아, 그러시구나. 그럼 저희 먼저 가 있을게요. | Oh, I see. Then we'll go on ahead. |  |  |
| 15 | A | 그래. 근데 서윤 씨, 점심은 먹었어? | Okay. By the way, Seoyun, have you had lunch? |  | **c02-t15** |
| 16 | B | 아직이요. 이것만 끝내고 먹으려고요. | Not yet. I was going to finish this first. |  |  |
| 17 | A | 그러지 말고 같이 가자. 내가 살게. | Forget that, come eat with me. My treat. |  |  |

### c02-t07 — `entity_consistency`, `lexical_consistency`

- 현재 발화 (B): 초록병 수정 시안은 오후에 올려 드리면 될까요?
- 정답: Would it be okay if I got you the revised Green Bottle mock-ups this afternoon?
- check `entity_consistency`: 초록병 is rendered as the client nickname 'Green Bottle' (e.g., 'the Green Bottle mock-ups'), not a literal green bottle and not a romanization like 'Chorokbyeong'.
- check `lexical_consistency`: 시안 is rendered as 'mock-up(s)' as established earlier (spelling variants 'mockups'/'mocks' are fine), not 'draft', 'design', 'proposal' or 'concept'.
- 문맥 효과: n=0 gives something like 'Can I upload the revised draft of the green bottle this afternoon?'. n=3 (turns 4-6) reaches 'mock-ups' but not the nickname; n=5 reaches turn 2 where 'Green Bottle' is introduced.
- 메모: Cue distances: 'mock-ups' 3 back (turn 4, also turn 3), 'Green Bottle' 5 back (turn 2). The request is polite (존댓말 to the boss) but that is visible without context, so register is not tagged here.

### c02-t10 — `fragment_incremental`, `omitted_argument`

- 현재 발화 (A): 여덟 시에 회사 앞에서 출발이야.
- 정답: leaves from in front of the office at eight.
- check `fragment_incremental`: Reads as the continuation of the previous fragment 'the bus for tomorrow's workshop…' (a predicate like '…leaves from in front of the office at eight', or a sentence with 'it/the bus' as subject).
- check `omitted_argument`: The omitted subject is the bus: no invented subject such as 'we', 'I', 'everyone' or 'the workshop starts'.
- 문맥 효과: n=0 invents a subject ('We leave from in front of the company at 8'). n=1 contains the fragment head '내일 워크숍 가는 버스' and should fix it.
- 메모: Streaming-ASR split: turn 9 has fragment_of_next=true. Cue distance 1.

### c02-t12 — `register_politeness`, `pronoun_coreference`

- 현재 발화 (B): 근데 팀장님도 같이 가세요?
- 정답: Are you coming with us too?
- check `register_politeness`: 팀장님 is handled as a respectful way of addressing the listener (English 'you', e.g., 'Are you coming with us too?' / 'Will you be joining us?'), not translated as a title such as 'Team Leader' or 'the team manager'.
- check `pronoun_coreference`: The question is about the listener ('are you coming'), not about a third person ('is the team leader / he / she going').
- 문맥 효과: n=0 reads 팀장님 as a third person: 'Is the team leader going too?'. n=1 has B saying '네, 팀장님' (turn 11) in the source, so SRC-based context should fix it; the English of turn 11 drops the title, so TGT-only context gives a weaker cue.
- 메모: Korean uses a job title as a second-person pronoun. Cue distance 1 (turn 11, source side only); turn 1 also has 팀장님.

### c02-t15 — `discourse_connective`, `context_trap`

- 현재 발화 (A): 그래. 근데 서윤 씨, 점심은 먹었어?
- 정답: Okay. By the way, Seoyun, have you had lunch?
- check `discourse_connective`: 근데 marks a topic shift ('By the way', 'Oh, and', 'Anyway') or is simply dropped; fails with a contrastive 'But'.
- check `context_trap`: Contains only the lunch question: does NOT bring in the workshop, the bus, the morning briefing or the mock-ups (e.g., no 'before the workshop', no 'before you go').
- 문맥 효과: n=0 tends to translate 근데 as 'But'. With context the topic shift from workshop logistics to lunch is visible (any n). Larger n adds more workshop/briefing content that a model might wrongly attach to the lunch question.
- 메모: Topic-change trap. The context_trap check passes at n=0 by design; it measures whether context makes the output worse.

## c03 — 동물병원 — 수의사(해요체)와 반려견 보호자(해요체), 기침하는 강아지 콩이

- **A**: 한지훈 (남, 30대, 수의사)
- **B**: 최미경 (여, 50대, 반려견 콩이(암컷)의 보호자)

| # | 화자 | ko | en | 조각 | 목표 |
|---|---|---|---|---|---|
| 0 | A | 안녕하세요. 오늘 콩이는 어디가 안 좋아서 왔어요? | Hi there. So what brings Kongi in today? |  |  |
| 1 | B | 며칠 전부터 기침을 자꾸 해서요. | Kongi's been coughing a lot the past few days. |  |  |
| 2 | B | 밤에 특히 심해요. | It gets really bad at night. |  |  |
| 3 | A | 예방접종은 다 돼 있죠? | Are all the vaccinations up to date? |  |  |
| 4 | B | 네, 다 했어요. | Yes, all done. |  |  |
| 5 | B | 제일 마지막 거는 지난달에 맞았고요. | Kongi got the last shot just last month. |  | **c03-t05** |
| 6 | A | 좋아요. 그럼 청진부터 해 볼게요. | Good. Let me take a listen to the chest first. |  |  |
| 7 | A | 폐 소리는 깨끗한데, 목이 좀 부었네요. | The lungs sound clear, but the throat's a little swollen. |  |  |
| 8 | B | 아이고, 우리 딸 어떡해. | Oh no, my poor girl. |  |  |
| 9 | A | 걱정 마세요, 심한 건 아니에요. | Don't worry, it's nothing serious. |  |  |
| 10 | A | 혹시 요즘 다른 강아지들 많은 데 간 적 있어요? | Have you been anywhere with a lot of other dogs lately? |  |  |
| 11 | B | 아, 주말에 애견 카페에 갔었어요. | Oh, we went to a dog café over the weekend. |  |  |
| 12 | B | 우리 딸이 거기서 다른 애들이랑 하루 종일 뛰어놀았거든요. | My girl spent the whole day running around with the other dogs there. |  | **c03-t12** |
| 13 | A | 아, 거기서 옮았을 수도 있겠네요. | Ah, she might have picked it up there. |  |  |
| 14 | A | 어머님, 오늘은 주사는 안 맞고 약만 먹으면 될 것 같아요. | Ma'am, I don't think she needs a shot today. Just some medicine should do it. |  | **c03-t14** |
| 15 | B | 네. 혹시 집에 있는 사람 감기약 먹여도 돼요? | Okay. Could I give her some of the human cold medicine we have at home? |  |  |
| 16 | A | 아뇨, 콩이 같은 소형견은 | No, for small dogs like Kongi, | → |  |
| 17 | A | 사람 약 먹으면 간이 상할 수 있어요. | human medicine can damage their liver. |  | **c03-t17** |

### c03-t05 — `word_sense`, `omitted_argument`

- 현재 발화 (B): 제일 마지막 거는 지난달에 맞았고요.
- 정답: Kongi got the last shot just last month.
- check `word_sense`: 맞았고요 is rendered as getting a vaccine shot ('got the last shot', 'had the last one'), not being hit, being right, or matching.
- check `omitted_argument`: If a recipient of the shot is expressed it is the dog (Kongi / the dog / she / it), not the speaker ('I got…'); a subjectless 'the last one was last month' is fine.
- 문맥 효과: n=0 reads 맞다 as 'hit' or 'correct', or makes the owner the subject ('I got the last one last month'). n=1 ('Yes, all done.') is not enough; n=3 reaches turn 3 ('vaccinations') and should fix both.
- 메모: Polysemy of 맞다 (be hit / be correct / fit / get a shot). Cue '예방접종' is 2 turns back (turn 3).

### c03-t12 — `lexical_consistency`, `word_sense`

- 현재 발화 (B): 우리 딸이 거기서 다른 애들이랑 하루 종일 뛰어놀았거든요.
- 정답: My girl spent the whole day running around with the other dogs there.
- check `lexical_consistency`: 우리 딸 is rendered as the dog, consistent with the earlier 'my poor girl' (e.g., 'my girl', 'my baby', 'Kongi', 'she'), not 'my daughter'.
- check `word_sense`: 애들 is rendered as the other dogs ('the other dogs', 'the other pups'), not 'kids' or 'children'.
- 문맥 효과: n=0 gives 'My daughter played with the other kids there all day.' n=1 ('dog café') should fix 애들; 우리 딸 = the dog needs n=5, which reaches turn 8 ('my poor girl').
- 메모: Korean pet owners call pets 우리 딸/아들 and other dogs 애들. Cue distances: 'dog café' 1 back, 'other dogs' 2 back, '우리 딸' 4 back (turn 8). Turn 10's English avoids naming Kongi so TGT n=3 does not tie the dog to 우리 딸 early.

### c03-t14 — `register_politeness`, `omitted_argument`

- 현재 발화 (A): 어머님, 오늘은 주사는 안 맞고 약만 먹으면 될 것 같아요.
- 정답: Ma'am, I don't think she needs a shot today. Just some medicine should do it.
- check `register_politeness`: 어머님 is rendered as a polite address to the client ('ma'am', or dropped), not 'Mother', 'Mom' or 'your mother'.
- check `omitted_argument`: The one who needs no shot and just medicine is the dog (she / it / Kongi), not the listener ('you don't need a shot').
- 문맥 효과: n=0 gives 'Mother, I don't think you need a shot today, just medicine.' n=1 ('she might have picked it up there', vet speaking about the dog) should fix both.
- 메모: Service-register address term 어머님 used by vets for any middle-aged female owner. Cue distance 1.

### c03-t17 — `fragment_incremental`, `omitted_argument`

- 현재 발화 (A): 사람 약 먹으면 간이 상할 수 있어요.
- 정답: human medicine can damage their liver.
- check `fragment_incremental`: Continues the previous fragment 'for small dogs like Kongi…': the liver damage is about those dogs taking human medicine (e.g., '…human medicine can damage their liver'), not a separate warning.
- check `omitted_argument`: The one taking the medicine and whose liver is at risk is the dog: 'their/her/its liver' or 'if you give them human medicine' are fine; fails with 'if you take…' or 'your liver'.
- 문맥 효과: n=0 turns it into a warning to the listener: 'If you take human medicine, it can damage your liver.' n=1 contains the fragment head '콩이 같은 소형견은' and should fix it.
- 메모: Streaming-ASR split: turn 16 has fragment_of_next=true. Cue distance 1.

## c04 — 가족 통화 — 추석에 기차 타고 내려가는 아들과 엄마 (반말)

- **A**: 김정희 (여, 50대, 도현의 엄마)
- **B**: 도현 (남, 20대 초반, 대학생 아들)

| # | 화자 | ko | en | 조각 | 목표 |
|---|---|---|---|---|---|
| 0 | B | 엄마, 나 방금 기차 탔어. | Mom, I just got on the train. |  |  |
| 1 | A | 그래, 몇 시에 도착해? | Okay, what time do you get in? |  |  |
| 2 | B | 여섯 시 반쯤. 아, 할머니 드릴 배도 한 박스 샀어. | Around 6:30. Oh, and I got a box of pears for Grandma. |  |  |
| 3 | A | 잘했네. 무겁지 않았어? | Good job. Wasn't it heavy? |  |  |
| 4 | B | 좀. 근데 역 계단에서 한 번 떨어뜨렸어. | A little. And I dropped it once on the station stairs. |  |  |
| 5 | A | 어머, 배 괜찮아? | Oh no, are the pears okay? |  | **c04-t05** |
| 6 | B | 몇 개 멍든 것 같은데 괜찮아. | I think a few got bruised, but it's fine. |  |  |
| 7 | B | 아, 근데 누나는 언제 와? | Oh, by the way, when's my sister coming? |  |  |
| 8 | A | 걔는 내일 아침에 온대. 회사 일이 아직 안 끝났대. | She's coming tomorrow morning. She says she's still not done with work. |  | **c04-t08** |
| 9 | A | 그래서 아빠가 내일 누나 데리러 역에 가기로 했어. | So Dad's picking your sister up at the station tomorrow. |  |  |
| 10 | B | 아빠 허리는 이제 괜찮아? | Is Dad's back okay now? |  |  |
| 11 | A | 그럭저럭. 너는 요즘 밥은 잘 챙겨 먹고 다니지? | So-so. And you're eating properly these days, right? |  |  |
| 12 | B | 그럼, 걱정 마. | Of course. Don't worry. |  | **c04-t12** |
| 13 | A | 걔도 맨날 그렇게 말하더라. | Your sister says that all the time too. |  | **c04-t13** |
| 14 | B | 누나는 진짜 안 챙겨 먹잖아. 나는 진짜 잘 먹어. | She really doesn't eat properly, though. I actually do. |  |  |
| 15 | A | 알았어. 도착하면 전화해. | Okay. Call me when you get in. |  |  |
| 16 | B | 응. 배도 조심해서 들고 갈게. | Okay. I'll be careful with the pears, too. |  |  |

### c04-t05 — `word_sense`

- 현재 발화 (A): 어머, 배 괜찮아?
- 정답: Oh no, are the pears okay?
- check `word_sense`: 배 is the pears ('are the pears okay?'), not the stomach/belly and not a boat.
- 문맥 효과: n=0 reads '배 괜찮아?' as 'Is your stomach okay?'. n=1 ('I dropped it once') suggests an object but not which; n=3 reaches turn 2 ('a box of pears').
- 메모: Polysemy of 배 (pear / stomach / boat). Cue distance 3 (turn 2); turn 4 mentions dropping 'it'.

### c04-t08 — `gender_reference`

- 현재 발화 (A): 걔는 내일 아침에 온대. 회사 일이 아직 안 끝났대.
- 정답: She's coming tomorrow morning. She says she's still not done with work.
- check `gender_reference`: Every pronoun for 걔 (the sister) is 'she/her' (the second clause may drop it, e.g. 'work isn't done yet'); fails with 'he' or 'they'.
- 문맥 효과: n=0 defaults to 'He's coming tomorrow morning.' n=1 has B asking about 누나 ('my sister') and should fix it.
- 메모: Gender cue distance 1 (turn 7, '누나').

### c04-t12 — `discourse_connective`

- 현재 발화 (B): 그럼, 걱정 마.
- 정답: Of course. Don't worry.
- check `discourse_connective`: 그럼 is rendered as an emphatic yes ('Of course', 'Sure', 'Yeah, definitely'), not 'Then' or 'So'.
- 문맥 효과: n=0 can read 그럼 as 'then' ('Then don't worry.'). n=1 (mom's yes/no question 'you're eating properly, right?') should fix it.
- 메모: 그럼 = 'then' vs 'of course'. Cue distance 1.

### c04-t13 — `gender_reference`, `context_trap`

- 현재 발화 (A): 걔도 맨날 그렇게 말하더라.
- 정답: Your sister says that all the time too.
- check `gender_reference`: 걔 is rendered as female ('she', 'your sister'); fails with 'he' or 'they'.
- check `context_trap`: Does not resolve 걔 to Dad, even though Dad is the most recently mentioned person (no 'Dad', 'your father', 'he').
- 문맥 효과: n=0 gives 'He always says that too.' n=1 ('Of course. Don't worry.') has no referent; n=3 (turns 10-12) contains Dad as the only third person, which invites 'Dad says that too'. n=5 reaches turn 9 ('your sister') and turn 8.
- 메모: Distractor: Dad (turns 9-10) is more recent than the sister. Gender cue '누나' is 4 turns back (turn 9); a mother would never call the father 걔.

## c05 — 동거 커플 — 집들이 계획 (반말, 여자친구가 남자친구를 '오빠'라고 부름)

- **A**: 민재 (남, 20대 후반, 회사원)
- **B**: 하린 (여, 20대 후반, 간호사, 민재의 여자친구)

| # | 화자 | ko | en | 조각 | 목표 |
|---|---|---|---|---|---|
| 0 | B | 오빠, 집들이 날짜 정했어? | Babe, did you pick a date for the housewarming? |  |  |
| 1 | A | 다음 주 토요일 어때? | How about next Saturday? |  |  |
| 2 | B | 좋아. 누구 부를까? | Works for me. Who should we invite? |  |  |
| 3 | A | 태오 형은 당연히 부르고. | Taeo, obviously. |  |  |
| 4 | B | 아, 박사님? 요즘 논문 때문에 바쁘다던데. | Oh, Doc? I heard he's swamped with his thesis these days. |  |  |
| 5 | A | 그래도 물어는 봐야지. | We should still ask him. |  |  |
| 6 | B | 나은 언니는 벌써 온다고 했어. | Na-eun already said she's coming. |  |  |
| 7 | A | 오, 빠르네. 음식은 어떡하지? | Oh, that was fast. What do we do about food? |  |  |
| 8 | B | 할매손 떡볶이 시키자. 거기 2인 세트 괜찮던데. | Let's order tteokbokki from Grandma's Hands. Their combo for two is pretty good. |  |  |
| 9 | A | 좋아. 아, 방금 박사님한테 문자 왔는데 못 온대. | Sounds good. Oh, Doc just texted me. He can't make it. |  | **c05-t09** |
| 10 | B | 아쉽다. 그럼 셋이네. | Aw, bummer. So it's just the three of us. |  |  |
| 11 | A | 아, 근데 그날 우리 엄마가, | Oh, but that day, my mom | → |  |
| 12 | A | 반찬 갖다주러 잠깐 들르신대. | says she's stopping by for a bit to bring us some side dishes. |  | **c05-t12** |
| 13 | B | 진짜? 그럼 할매손 세트 하나 더 시키자. | Really? Then let's get one more combo from Grandma's Hands. |  | **c05-t13** |
| 14 | B | 오빠가 어머님한테 뭐 드시고 싶은지 물어봐 줘. | Can you ask your mom what she'd like to eat? |  | **c05-t14** |
| 15 | A | 알았어, 물어볼게. | Okay, I'll ask her. |  |  |
| 16 | B | 아 맞다, 그리고 휴지도 사야 돼. | Oh, right, and we need to buy toilet paper. |  |  |
| 17 | A | 그건 내가 퇴근길에 살게. | I'll grab that on my way home from work. |  |  |

### c05-t09 — `entity_consistency`, `context_trap`

- 현재 발화 (A): 좋아. 아, 방금 박사님한테 문자 왔는데 못 온대.
- 정답: Sounds good. Oh, Doc just texted me. He can't make it.
- check `entity_consistency`: 박사님 is rendered as the nickname 'Doc' (or 'Taeo', or an equivalent that clearly reads as the friend's nickname), not 'the doctor', 'the professor' or 'Dr. …'.
- check `context_trap`: Does NOT add a reason for not coming (e.g., his thesis or being busy); the utterance gives none.
- 문맥 효과: n=0 gives 'I just got a text from the doctor, he says he can't come.' The nickname is introduced 5 turns back (turn 4), so only n=5 fixes it, and n=5 is also the first window that contains the thesis, which invites an added 'because of his thesis'.
- 메모: Nickname cue distance 5 (turn 4, '박사님' for Taeo). The trap and the fix arrive in the same window on purpose.

### c05-t12 — `fragment_incremental`, `gender_reference`

- 현재 발화 (A): 반찬 갖다주러 잠깐 들르신대.
- 정답: says she's stopping by for a bit to bring us some side dishes.
- check `fragment_incremental`: Continues 'my mom…': the one dropping by is the mom (a predicate continuation like '…says she's stopping by', or a sentence with 'she'), not the speaker or listener ('I'll stop by', 'you').
- check `gender_reference`: Any pronoun for the visitor is 'she'; fails with 'he' or 'they'.
- 문맥 효과: n=0 invents a subject and gender: 'He says he'll drop by briefly to bring side dishes.' n=1 contains '우리 엄마가' and should fix both.
- 메모: Streaming-ASR split: turn 11 has fragment_of_next=true. Gender cue distance 1.

### c05-t13 — `entity_consistency`, `lexical_consistency`

- 현재 발화 (B): 진짜? 그럼 할매손 세트 하나 더 시키자.
- 정답: Really? Then let's get one more combo from Grandma's Hands.
- check `entity_consistency`: 할매손 is rendered as 'Grandma's Hands' (the restaurant name used earlier; a minor variant like 'Grandma's Hand' used as the name is fine), not a romanization ('Halmaeson') or a common-noun 'grandma's hand'.
- check `lexical_consistency`: 세트 is rendered as 'combo' as established earlier, not 'set' or 'set meal'.
- 문맥 효과: n=0 gives 'Then let's order one more set from Halmaeson.' Both terms were introduced 5 turns back (turn 8), so n=1 and n=3 miss them; n=5 should fix both.
- 메모: Cue distance 5 (turn 8).

### c05-t14 — `register_politeness`, `pronoun_coreference`

- 현재 발화 (B): 오빠가 어머님한테 뭐 드시고 싶은지 물어봐 줘.
- 정답: Can you ask your mom what she'd like to eat?
- check `register_politeness`: Address terms fit a girlfriend talking to her boyfriend: 오빠 becomes 'you' (optionally 'babe'), not 'Oppa' or 'my brother'; 어머님 becomes 'your mom', not 'Mother'.
- check `pronoun_coreference`: The mother is the listener's mom ('your mom'), not the speaker's ('my mom') and not an unspecified 'Mother'.
- 문맥 효과: n=0 gives 'Oppa, please ask Mother what she wants to eat' or reads 오빠 as a brother. n=3 reaches turn 11 where A says 'my mom', which makes 어머님 = 'your mom'.
- 메모: Kinship/title terms used as second-person address and as in-law honorific. Cue distance 2-3 (turns 11-12).

