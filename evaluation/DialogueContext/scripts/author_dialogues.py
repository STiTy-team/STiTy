"""Author the 5 ko/en dialogues and 20 context-dependent instances for DialogueContext.

    python evaluation/DialogueContext/scripts/author_dialogues.py

Writes data/dialogues.jsonl, data/instances.jsonl, data/preview.md (schemas in DESIGN.md)
and asserts the dataset contract before writing anything.
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

TAGS = (
    "pronoun_coreference", "omitted_argument", "gender_reference", "register_politeness",
    "lexical_consistency", "word_sense", "fragment_incremental", "discourse_connective",
    "entity_consistency", "context_trap",
)
MIN_PREVIOUS = 5
MIN_PER_TAG = 3

F = True  # fragment_of_next marker, for readability in the turn tables

# Each turn: (speaker, ko, en) or (speaker, ko, en, F) when the same speaker keeps going.
DIALOGUES = [
{
    "dialogue_id": "c01",
    "setting": "친구 사이 수다 — 여자친구 생일 선물을 두고 나온 이야기 (반말)",
    "speakers": {"A": "준호 (남, 20대 후반, 회사원)", "B": "다은 (여, 20대 후반, 준호의 대학 친구)"},
    "turns": [
        ("A", "야, 나 어제 진짜 정신없었어.", "Man, yesterday was total chaos."),
        ("B", "왜, 무슨 일 있었어?", "Why, what happened?"),
        ("A", "수진이 생일이라 저녁 예약해 놨었거든.", "It was Sujin's birthday, so I'd booked us a dinner."),
        ("A", "근데 식당 가는 길에 딱 생각난 거야,", "But on the way to the restaurant, it suddenly hit me,", F),
        ("A", "여자친구 주려고 산 양말을 집에 두고 왔다는 게.", "I'd left the socks I bought for my girlfriend at home."),
        ("B", "양말? 선물로 양말 산 거야?", "Socks? You got socks as a present?"),
        ("A", "그냥 그런 거 아니고, 걔가 옛날부터 갖고 싶어 하던 거야.", "Not just any socks. They're the ones she's wanted forever."),
        ("B", "그래서 어떻게 했어?", "So what'd you do?"),
        ("A", "당연히 다시 가지러 갔지.", "I went back for them, obviously."),
        ("A", "근데 그러고 나니까 시간이 너무 빠듯해서,", "But after that I was cutting it really close, so", F),
        ("A", "결국 택시를 탔어.", "I ended up taking a taxi."),
        ("B", "택시 탔으면 여자친구 안 기다리게 했겠네.", "Well, at least with a taxi you didn't keep your girlfriend waiting."),
        ("A", "그렇기는 한데 결국 늦었어.", "You'd think so, but I was still late."),
        ("A", "기사님이 골목길로 막 돌아가 주셨는데도 길이 너무 막혔어.", "The driver kept cutting through back alleys and everything, but traffic was just awful."),
        ("B", "아이고. 그래서 걔는 좋아했어?", "Oh no. So did she like them?"),
        ("A", "응, 바로 신어 보더니 딱 맞는다고 엄청 좋아하더라.", "Yeah, she put them on right away, said they fit perfectly, and totally loved them."),
        ("B", "다행이다. 다음엔 전날 미리 챙겨 놔.", "Phew. Next time, get them ready the night before."),
    ],
},
{
    "dialogue_id": "c02",
    "setting": "직장 — 팀장(반말)과 사원(존댓말), 거래처 배너 시안과 워크숍 이야기",
    "speakers": {"A": "박준영 팀장 (남, 40대, 마케팅팀 팀장)", "B": "이서윤 (여, 20대 후반, 마케팅팀 사원)"},
    "turns": [
        ("A", "서윤 씨, 잠깐 시간 돼?", "Seoyun, got a sec?"),
        ("B", "네, 팀장님. 무슨 일이세요?", "Yes, of course. What is it?"),
        ("A", "우리가 초록병이라고 부르는 음료 회사 있잖아.", "You know that beverage company we call Green Bottle?"),
        ("A", "거기서 배너 시안이 다 마음에 안 든대.", "They say they don't like any of the banner mock-ups."),
        ("B", "네? 어제 시안 세 개 다 보내 드렸는데요.", "Sorry? I sent over all three mock-ups yesterday."),
        ("A", "응, 셋 다 색이 너무 튄다고.", "Yeah, they said the colors on all three are too loud."),
        ("B", "아... 그럼 톤을 좀 차분하게 다시 잡아 볼게요.", "Oh... Then I'll rework them in a calmer palette."),
        ("B", "초록병 수정 시안은 오후에 올려 드리면 될까요?", "Would it be okay if I got you the revised Green Bottle mock-ups this afternoon?"),
        ("A", "그래, 네 시 전에만 줘.", "Sure, just get them to me before four."),
        ("A", "아, 그리고 내일 워크숍 가는 버스,", "Oh, and the bus for tomorrow's workshop", F),
        ("A", "여덟 시에 회사 앞에서 출발이야.", "leaves from in front of the office at eight."),
        ("B", "네, 팀장님. 공지 봤습니다.", "Yes, I saw the announcement."),
        ("B", "근데 팀장님도 같이 가세요?", "Are you coming with us too?"),
        ("A", "나? 나는 오후에 합류해. 오전에 본부장님 보고가 있거든.", "Me? I'll join in the afternoon. I've got a briefing with the division head in the morning."),
        ("B", "아, 그러시구나. 그럼 저희 먼저 가 있을게요.", "Oh, I see. Then we'll go on ahead."),
        ("A", "그래. 근데 서윤 씨, 점심은 먹었어?", "Okay. By the way, Seoyun, have you had lunch?"),
        ("B", "아직이요. 이것만 끝내고 먹으려고요.", "Not yet. I was going to finish this first."),
        ("A", "그러지 말고 같이 가자. 내가 살게.", "Forget that, come eat with me. My treat."),
    ],
},
{
    "dialogue_id": "c03",
    "setting": "동물병원 — 수의사(해요체)와 반려견 보호자(해요체), 기침하는 강아지 콩이",
    "speakers": {"A": "한지훈 (남, 30대, 수의사)", "B": "최미경 (여, 50대, 반려견 콩이(암컷)의 보호자)"},
    "turns": [
        ("A", "안녕하세요. 오늘 콩이는 어디가 안 좋아서 왔어요?", "Hi there. So what brings Kongi in today?"),
        ("B", "며칠 전부터 기침을 자꾸 해서요.", "Kongi's been coughing a lot the past few days."),
        ("B", "밤에 특히 심해요.", "It gets really bad at night."),
        ("A", "예방접종은 다 돼 있죠?", "Are all the vaccinations up to date?"),
        ("B", "네, 다 했어요.", "Yes, all done."),
        ("B", "제일 마지막 거는 지난달에 맞았고요.", "Kongi got the last shot just last month."),
        ("A", "좋아요. 그럼 청진부터 해 볼게요.", "Good. Let me take a listen to the chest first."),
        ("A", "폐 소리는 깨끗한데, 목이 좀 부었네요.", "The lungs sound clear, but the throat's a little swollen."),
        ("B", "아이고, 우리 딸 어떡해.", "Oh no, my poor girl."),
        ("A", "걱정 마세요, 심한 건 아니에요.", "Don't worry, it's nothing serious."),
        ("A", "혹시 요즘 다른 강아지들 많은 데 간 적 있어요?", "Have you been anywhere with a lot of other dogs lately?"),
        ("B", "아, 주말에 애견 카페에 갔었어요.", "Oh, we went to a dog café over the weekend."),
        ("B", "우리 딸이 거기서 다른 애들이랑 하루 종일 뛰어놀았거든요.", "My girl spent the whole day running around with the other dogs there."),
        ("A", "아, 거기서 옮았을 수도 있겠네요.", "Ah, she might have picked it up there."),
        ("A", "어머님, 오늘은 주사는 안 맞고 약만 먹으면 될 것 같아요.", "Ma'am, I don't think she needs a shot today. Just some medicine should do it."),
        ("B", "네. 혹시 집에 있는 사람 감기약 먹여도 돼요?", "Okay. Could I give her some of the human cold medicine we have at home?"),
        ("A", "아뇨, 콩이 같은 소형견은", "No, for small dogs like Kongi,", F),
        ("A", "사람 약 먹으면 간이 상할 수 있어요.", "human medicine can damage their liver."),
    ],
},
{
    "dialogue_id": "c04",
    "setting": "가족 통화 — 추석에 기차 타고 내려가는 아들과 엄마 (반말)",
    "speakers": {"A": "김정희 (여, 50대, 도현의 엄마)", "B": "도현 (남, 20대 초반, 대학생 아들)"},
    "turns": [
        ("B", "엄마, 나 방금 기차 탔어.", "Mom, I just got on the train."),
        ("A", "그래, 몇 시에 도착해?", "Okay, what time do you get in?"),
        ("B", "여섯 시 반쯤. 아, 할머니 드릴 배도 한 박스 샀어.", "Around 6:30. Oh, and I got a box of pears for Grandma."),
        ("A", "잘했네. 무겁지 않았어?", "Good job. Wasn't it heavy?"),
        ("B", "좀. 근데 역 계단에서 한 번 떨어뜨렸어.", "A little. And I dropped it once on the station stairs."),
        ("A", "어머, 배 괜찮아?", "Oh no, are the pears okay?"),
        ("B", "몇 개 멍든 것 같은데 괜찮아.", "I think a few got bruised, but it's fine."),
        ("B", "아, 근데 누나는 언제 와?", "Oh, by the way, when's my sister coming?"),
        ("A", "걔는 내일 아침에 온대. 회사 일이 아직 안 끝났대.", "She's coming tomorrow morning. She says she's still not done with work."),
        ("A", "그래서 아빠가 내일 누나 데리러 역에 가기로 했어.", "So Dad's picking your sister up at the station tomorrow."),
        ("B", "아빠 허리는 이제 괜찮아?", "Is Dad's back okay now?"),
        ("A", "그럭저럭. 너는 요즘 밥은 잘 챙겨 먹고 다니지?", "So-so. And you're eating properly these days, right?"),
        ("B", "그럼, 걱정 마.", "Of course. Don't worry."),
        ("A", "걔도 맨날 그렇게 말하더라.", "Your sister says that all the time too."),
        ("B", "누나는 진짜 안 챙겨 먹잖아. 나는 진짜 잘 먹어.", "She really doesn't eat properly, though. I actually do."),
        ("A", "알았어. 도착하면 전화해.", "Okay. Call me when you get in."),
        ("B", "응. 배도 조심해서 들고 갈게.", "Okay. I'll be careful with the pears, too."),
    ],
},
{
    "dialogue_id": "c05",
    "setting": "동거 커플 — 집들이 계획 (반말, 여자친구가 남자친구를 '오빠'라고 부름)",
    "speakers": {"A": "민재 (남, 20대 후반, 회사원)", "B": "하린 (여, 20대 후반, 간호사, 민재의 여자친구)"},
    "turns": [
        ("B", "오빠, 집들이 날짜 정했어?", "Babe, did you pick a date for the housewarming?"),
        ("A", "다음 주 토요일 어때?", "How about next Saturday?"),
        ("B", "좋아. 누구 부를까?", "Works for me. Who should we invite?"),
        ("A", "태오 형은 당연히 부르고.", "Taeo, obviously."),
        ("B", "아, 박사님? 요즘 논문 때문에 바쁘다던데.", "Oh, Doc? I heard he's swamped with his thesis these days."),
        ("A", "그래도 물어는 봐야지.", "We should still ask him."),
        ("B", "나은 언니는 벌써 온다고 했어.", "Na-eun already said she's coming."),
        ("A", "오, 빠르네. 음식은 어떡하지?", "Oh, that was fast. What do we do about food?"),
        ("B", "할매손 떡볶이 시키자. 거기 2인 세트 괜찮던데.", "Let's order tteokbokki from Grandma's Hands. Their combo for two is pretty good."),
        ("A", "좋아. 아, 방금 박사님한테 문자 왔는데 못 온대.", "Sounds good. Oh, Doc just texted me. He can't make it."),
        ("B", "아쉽다. 그럼 셋이네.", "Aw, bummer. So it's just the three of us."),
        ("A", "아, 근데 그날 우리 엄마가,", "Oh, but that day, my mom", F),
        ("A", "반찬 갖다주러 잠깐 들르신대.", "says she's stopping by for a bit to bring us some side dishes."),
        ("B", "진짜? 그럼 할매손 세트 하나 더 시키자.", "Really? Then let's get one more combo from Grandma's Hands."),
        ("B", "오빠가 어머님한테 뭐 드시고 싶은지 물어봐 줘.", "Can you ask your mom what she'd like to eat?"),
        ("A", "알았어, 물어볼게.", "Okay, I'll ask her."),
        ("B", "아 맞다, 그리고 휴지도 사야 돼.", "Oh, right, and we need to buy toilet paper."),
        ("A", "그건 내가 퇴근길에 살게.", "I'll grab that on my way home from work."),
    ],
},
]

# Each instance: (dialogue_id, target_turn_id, [tags], [(tag, requirement)], expected_context_effect, note)
INSTANCES = [
# ---------------------------------------------------------------- c01
("c01", 6, ["gender_reference", "pronoun_coreference"],
 [("gender_reference", "Refers to 걔 (the girlfriend, Sujin) as 'she/her'; fails with 'he', 'they' or 'it'."),
  ("pronoun_coreference", "거 is resolved to the socks: plural 'they/the ones/socks' or 'a pair', not a singular 'it', 'something' or 'the thing' standing for them.")],
 "n=0 has neither referent: typically 'it's something he's wanted for a long time'. n=1 (B: 'Socks?') fixes the object but not the gender. n=3 reaches turn 4 ('my girlfriend') and should fix both.",
 "Gender cue '여자친구' is 2 turns back (turn 4); 'socks' is 1 back. The current turn deliberately avoids the word 양말."),
("c01", 8, ["omitted_argument"],
 [("omitted_argument", "The omitted object is the socks: 'them' or 'the socks'; fails with 'it', 'something' or no object that makes sense.")],
 "n=0 gives 'I went back to get it.' The socks are 2-3 turns back (turns 5-6), so n=1 ('So what'd you do?') still misses; n=3 should fix.",
 "STiTy-style example adapted from '그래서 다시 가지러 갔지.' Object cue distance 2-3 (turns 5-6)."),
("c01", 12, ["discourse_connective"],
 [("discourse_connective", "그렇기는 한데 is rendered as conceding B's assumption while contradicting it ('You'd think so, but…', 'It should have, but…', 'That was the idea, but…'); a hedged 'Yeah, but I was still late' is also fine. Fails only with an explicit affirmation of B's claim ('That's true, but…', 'You're right, but…') that then contradicts itself.")],
 "n=0 cannot know what is being conceded and usually outputs 'That's true, but I ended up being late.' n=1 (B: 'at least with a taxi you didn't keep your girlfriend waiting') is enough to pick the counter-expectation reading.",
 "STiTy-style example adapted from prev '그래서 택시를 탔어.' → '그렇기는 한데 결국 늦었어.' Cue distance 1 (turn 11), taxi at 2."),
("c01", 14, ["gender_reference", "pronoun_coreference", "context_trap"],
 [("gender_reference", "Refers to 걔 as female ('she/her', 'your girlfriend' or 'Sujin'); fails with 'he' or 'they'."),
  ("pronoun_coreference", "걔 refers to the girlfriend (e.g., 'did she like them', 'was your girlfriend happy'), not to the taxi driver."),
  ("context_trap", "Does NOT pull in the driver or the lateness (no 'even though you were late', no mention of the driver or traffic).")],
 "n=0 defaults to 'did he like it?'. n=1 shows only the driver (turn 13), the most recent person, which invites 'he' = the driver. n=3 reaches turn 11 ('your girlfriend') and should fix the gender and referent.",
 "Distractor: the taxi driver is the most recently mentioned person. Gender cue '여자친구' is 3 turns back (turn 11)."),
# ---------------------------------------------------------------- c02
("c02", 7, ["entity_consistency", "lexical_consistency"],
 [("entity_consistency", "초록병 is rendered as the client nickname 'Green Bottle' (e.g., 'the Green Bottle mock-ups'), not a literal green bottle and not a romanization like 'Chorokbyeong'."),
  ("lexical_consistency", "시안 is rendered as 'mock-up(s)' as established earlier (spelling variants 'mockups'/'mocks' are fine), not 'draft', 'design', 'proposal' or 'concept'.")],
 "n=0 gives something like 'Can I upload the revised draft of the green bottle this afternoon?'. n=3 (turns 4-6) reaches 'mock-ups' but not the nickname; n=5 reaches turn 2 where 'Green Bottle' is introduced.",
 "Cue distances: 'mock-ups' 3 back (turn 4, also turn 3), 'Green Bottle' 5 back (turn 2). The request is polite (존댓말 to the boss) but that is visible without context, so register is not tagged here."),
("c02", 10, ["fragment_incremental", "omitted_argument"],
 [("fragment_incremental", "Reads as the continuation of the previous fragment 'the bus for tomorrow's workshop…' (a predicate like '…leaves from in front of the office at eight', or a sentence with 'it/the bus' as subject)."),
  ("omitted_argument", "The omitted subject is the bus: no invented subject such as 'we', 'I', 'everyone' or 'the workshop starts'.")],
 "n=0 invents a subject ('We leave from in front of the company at 8'). n=1 contains the fragment head '내일 워크숍 가는 버스' and should fix it.",
 "Streaming-ASR split: turn 9 has fragment_of_next=true. Cue distance 1."),
("c02", 12, ["register_politeness", "pronoun_coreference"],
 [("register_politeness", "팀장님 is handled as a respectful way of addressing the listener (English 'you', e.g., 'Are you coming with us too?' / 'Will you be joining us?'), not translated as a title such as 'Team Leader' or 'the team manager'."),
  ("pronoun_coreference", "The question is about the listener ('are you coming'), not about a third person ('is the team leader / he / she going').")],
 "n=0 reads 팀장님 as a third person: 'Is the team leader going too?'. n=1 has B saying '네, 팀장님' (turn 11) in the source, so SRC-based context should fix it; the English of turn 11 drops the title, so TGT-only context gives a weaker cue.",
 "Korean uses a job title as a second-person pronoun. Cue distance 1 (turn 11, source side only); turn 1 also has 팀장님."),
("c02", 15, ["discourse_connective", "context_trap"],
 [("discourse_connective", "근데 marks a topic shift ('By the way', 'Oh, and', 'Anyway') or is simply dropped; fails with a contrastive 'But'.",),
  ("context_trap", "Contains only the lunch question: does NOT bring in the workshop, the bus, the morning briefing or the mock-ups (e.g., no 'before the workshop', no 'before you go').")],
 "n=0 tends to translate 근데 as 'But'. With context the topic shift from workshop logistics to lunch is visible (any n). Larger n adds more workshop/briefing content that a model might wrongly attach to the lunch question.",
 "Topic-change trap. The context_trap check passes at n=0 by design; it measures whether context makes the output worse."),
# ---------------------------------------------------------------- c03
("c03", 5, ["word_sense", "omitted_argument"],
 [("word_sense", "맞았고요 is rendered as getting a vaccine shot ('got the last shot', 'had the last one'), not being hit, being right, or matching."),
  ("omitted_argument", "If a recipient of the shot is expressed it is the dog (Kongi / the dog / she / it), not the speaker ('I got…'); a subjectless 'the last one was last month' is fine.")],
 "n=0 reads 맞다 as 'hit' or 'correct', or makes the owner the subject ('I got the last one last month'). n=1 ('Yes, all done.') is not enough; n=3 reaches turn 3 ('vaccinations') and should fix both.",
 "Polysemy of 맞다 (be hit / be correct / fit / get a shot). Cue '예방접종' is 2 turns back (turn 3)."),
("c03", 12, ["lexical_consistency", "word_sense"],
 [("lexical_consistency", "우리 딸 is rendered as the dog, consistent with the earlier 'my poor girl' (e.g., 'my girl', 'my baby', 'Kongi', 'she'), not 'my daughter'."),
  ("word_sense", "애들 is rendered as the other dogs ('the other dogs', 'the other pups'), not 'kids' or 'children'.")],
 "n=0 gives 'My daughter played with the other kids there all day.' n=1 ('dog café') should fix 애들; 우리 딸 = the dog needs n=5, which reaches turn 8 ('my poor girl').",
 "Korean pet owners call pets 우리 딸/아들 and other dogs 애들. Cue distances: 'dog café' 1 back, 'other dogs' 2 back, '우리 딸' 4 back (turn 8). Turn 10's English avoids naming Kongi so TGT n=3 does not tie the dog to 우리 딸 early."),
("c03", 14, ["register_politeness", "omitted_argument"],
 [("register_politeness", "어머님 is rendered as a polite address to the client ('ma'am', or dropped), not 'Mother', 'Mom' or 'your mother'."),
  ("omitted_argument", "The one who needs no shot and just medicine is the dog (she / it / Kongi), not the listener ('you don't need a shot').")],
 "n=0 gives 'Mother, I don't think you need a shot today, just medicine.' n=1 ('she might have picked it up there', vet speaking about the dog) should fix both.",
 "Service-register address term 어머님 used by vets for any middle-aged female owner. Cue distance 1."),
("c03", 17, ["fragment_incremental", "omitted_argument"],
 [("fragment_incremental", "Continues the previous fragment 'for small dogs like Kongi…': the liver damage is about those dogs taking human medicine (e.g., '…human medicine can damage their liver'), not a separate warning."),
  ("omitted_argument", "The one taking the medicine and whose liver is at risk is the dog: 'their/her/its liver' or 'if you give them human medicine' are fine; fails with 'if you take…' or 'your liver'.")],
 "n=0 turns it into a warning to the listener: 'If you take human medicine, it can damage your liver.' n=1 contains the fragment head '콩이 같은 소형견은' and should fix it.",
 "Streaming-ASR split: turn 16 has fragment_of_next=true. Cue distance 1."),
# ---------------------------------------------------------------- c04
("c04", 5, ["word_sense"],
 [("word_sense", "배 is the pears ('are the pears okay?'), not the stomach/belly and not a boat.")],
 "n=0 reads '배 괜찮아?' as 'Is your stomach okay?'. n=1 ('I dropped it once') suggests an object but not which; n=3 reaches turn 2 ('a box of pears').",
 "Polysemy of 배 (pear / stomach / boat). Cue distance 3 (turn 2); turn 4 mentions dropping 'it'."),
("c04", 8, ["gender_reference"],
 [("gender_reference", "Every pronoun for 걔 (the sister) is 'she/her' (the second clause may drop it, e.g. 'work isn't done yet'); fails with 'he' or 'they'.")],
 "n=0 defaults to 'He's coming tomorrow morning.' n=1 has B asking about 누나 ('my sister') and should fix it.",
 "Gender cue distance 1 (turn 7, '누나')."),
("c04", 12, ["discourse_connective"],
 [("discourse_connective", "그럼 is rendered as an emphatic yes ('Of course', 'Sure', 'Yeah, definitely'), not 'Then' or 'So'.")],
 "n=0 can read 그럼 as 'then' ('Then don't worry.'). n=1 (mom's yes/no question 'you're eating properly, right?') should fix it.",
 "그럼 = 'then' vs 'of course'. Cue distance 1."),
("c04", 13, ["gender_reference", "context_trap"],
 [("gender_reference", "걔 is rendered as female ('she', 'your sister'); fails with 'he' or 'they'."),
  ("context_trap", "Does not resolve 걔 to Dad, even though Dad is the most recently mentioned person (no 'Dad', 'your father', 'he').")],
 "n=0 gives 'He always says that too.' n=1 ('Of course. Don't worry.') has no referent; n=3 (turns 10-12) contains Dad as the only third person, which invites 'Dad says that too'. n=5 reaches turn 9 ('your sister') and turn 8.",
 "Distractor: Dad (turns 9-10) is more recent than the sister. Gender cue '누나' is 4 turns back (turn 9); a mother would never call the father 걔."),
# ---------------------------------------------------------------- c05
("c05", 9, ["entity_consistency", "context_trap"],
 [("entity_consistency", "박사님 is rendered as the nickname 'Doc' (or 'Taeo', or an equivalent that clearly reads as the friend's nickname), not 'the doctor', 'the professor' or 'Dr. …'."),
  ("context_trap", "Does NOT add a reason for not coming (e.g., his thesis or being busy); the utterance gives none.")],
 "n=0 gives 'I just got a text from the doctor, he says he can't come.' The nickname is introduced 5 turns back (turn 4), so only n=5 fixes it, and n=5 is also the first window that contains the thesis, which invites an added 'because of his thesis'.",
 "Nickname cue distance 5 (turn 4, '박사님' for Taeo). The trap and the fix arrive in the same window on purpose."),
("c05", 12, ["fragment_incremental", "gender_reference"],
 [("fragment_incremental", "Continues 'my mom…': the one dropping by is the mom (a predicate continuation like '…says she's stopping by', or a sentence with 'she'), not the speaker or listener ('I'll stop by', 'you')."),
  ("gender_reference", "Any pronoun for the visitor is 'she'; fails with 'he' or 'they'.")],
 "n=0 invents a subject and gender: 'He says he'll drop by briefly to bring side dishes.' n=1 contains '우리 엄마가' and should fix both.",
 "Streaming-ASR split: turn 11 has fragment_of_next=true. Gender cue distance 1."),
("c05", 13, ["entity_consistency", "lexical_consistency"],
 [("entity_consistency", "할매손 is rendered as 'Grandma's Hands' (the restaurant name used earlier; a minor variant like 'Grandma's Hand' used as the name is fine), not a romanization ('Halmaeson') or a common-noun 'grandma's hand'."),
  ("lexical_consistency", "세트 is rendered as 'combo' as established earlier, not 'set' or 'set meal'.")],
 "n=0 gives 'Then let's order one more set from Halmaeson.' Both terms were introduced 5 turns back (turn 8), so n=1 and n=3 miss them; n=5 should fix both.",
 "Cue distance 5 (turn 8)."),
("c05", 14, ["register_politeness", "pronoun_coreference"],
 [("register_politeness", "Address terms fit a girlfriend talking to her boyfriend: 오빠 becomes 'you' (optionally 'babe'), not 'Oppa' or 'my brother'; 어머님 becomes 'your mom', not 'Mother'."),
  ("pronoun_coreference", "The mother is the listener's mom ('your mom'), not the speaker's ('my mom') and not an unspecified 'Mother'.")],
 "n=0 gives 'Oppa, please ask Mother what she wants to eat' or reads 오빠 as a brother. n=3 reaches turn 11 where A says 'my mom', which makes 어머님 = 'your mom'.",
 "Kinship/title terms used as second-person address and as in-law honorific. Cue distance 2-3 (turns 11-12)."),
]


def build():
    dialogues, turn_index = [], {}
    for d in DIALOGUES:
        for turn_id, t in enumerate(d["turns"]):
            speaker, ko, en = t[:3]
            frag = len(t) == 4 and t[3] is True
            row = {"dialogue_id": d["dialogue_id"], "turn_id": turn_id, "speaker": speaker,
                   "ko": ko, "en": en, "fragment_of_next": frag}
            dialogues.append(row)
            turn_index[(d["dialogue_id"], turn_id)] = row
    meta = {d["dialogue_id"]: d for d in DIALOGUES}

    instances = []
    for dialogue_id, target, tags, checks, effect, note in INSTANCES:
        cur = turn_index[(dialogue_id, target)]
        previous = [{"turn_id": r["turn_id"], "speaker": r["speaker"], "ko": r["ko"], "en": r["en"]}
                    for r in dialogues if r["dialogue_id"] == dialogue_id and r["turn_id"] < target]
        instances.append({
            "instance_id": f"{dialogue_id}-t{target:02d}", "dialogue_id": dialogue_id,
            "target_turn_id": target, "source_language": "ko", "target_language": "en",
            "speakers": meta[dialogue_id]["speakers"], "current_speaker": cur["speaker"],
            "previous_turns": previous, "current_source": cur["ko"],
            "reference_translation": cur["en"], "challenge_tags": tags,
            "checks": [{"tag": tag, "requirement": req} for tag, req in checks],
            "expected_context_effect": effect, "note": note,
        })
    return dialogues, instances


def validate(dialogues, instances):
    by_dialogue = {}
    for r in dialogues:
        assert set(r) == {"dialogue_id", "turn_id", "speaker", "ko", "en", "fragment_of_next"}, r
        assert r["speaker"] in ("A", "B") and r["ko"].strip() and r["en"].strip(), r
        by_dialogue.setdefault(r["dialogue_id"], []).append(r)
    assert len(by_dialogue) == 5, "expected 5 dialogues"
    for did, rows in by_dialogue.items():
        assert [r["turn_id"] for r in rows] == list(range(len(rows))), did
        assert 14 <= len(rows) <= 18, (did, len(rows))
        assert rows[-1]["fragment_of_next"] is False, did
        for a, b in zip(rows, rows[1:]):
            if a["fragment_of_next"]:
                assert a["speaker"] == b["speaker"], (did, a["turn_id"], "fragment continues with other speaker")
        assert sum(r["fragment_of_next"] for r in rows) >= 1 or did == "c04", did

    keys = ["instance_id", "dialogue_id", "target_turn_id", "source_language", "target_language",
            "speakers", "current_speaker", "previous_turns", "current_source", "reference_translation",
            "challenge_tags", "checks", "expected_context_effect", "note"]
    assert len(instances) == 20, len(instances)
    assert len({i["instance_id"] for i in instances}) == 20
    tag_count = Counter()
    for inst in instances:
        assert list(inst) == keys, inst["instance_id"]
        rows = by_dialogue[inst["dialogue_id"]]
        cur = rows[inst["target_turn_id"]]
        assert inst["current_source"] == cur["ko"] and inst["reference_translation"] == cur["en"]
        assert inst["current_speaker"] == cur["speaker"]
        expect_prev = [{"turn_id": r["turn_id"], "speaker": r["speaker"], "ko": r["ko"], "en": r["en"]}
                       for r in rows[:inst["target_turn_id"]]]
        assert inst["previous_turns"] == expect_prev, inst["instance_id"]
        assert len(inst["previous_turns"]) >= MIN_PREVIOUS, inst["instance_id"]
        assert set(inst["speakers"]) == {r["speaker"] for r in rows}, inst["instance_id"]
        tags = inst["challenge_tags"]
        assert tags and len(set(tags)) == len(tags) and set(tags) <= set(TAGS), inst["instance_id"]
        assert 1 <= len(inst["checks"]) <= 3, inst["instance_id"]
        check_tags = [c["tag"] for c in inst["checks"]]
        assert set(check_tags) <= set(tags), (inst["instance_id"], "check tag not in challenge_tags")
        assert set(tags) <= set(check_tags), (inst["instance_id"], "challenge tag without a check")
        assert all(c["requirement"].strip() for c in inst["checks"])
        assert inst["expected_context_effect"].strip() and inst["note"].strip()
        tag_count.update(tags)
    per_dialogue = Counter(i["dialogue_id"] for i in instances)
    assert all(v == 4 for v in per_dialogue.values()) and len(per_dialogue) == 5, per_dialogue
    for tag in TAGS:
        assert tag_count[tag] >= MIN_PER_TAG, (tag, tag_count[tag])
    return tag_count


def md_cell(s):
    return s.replace("|", "\\|")


def preview(dialogues, instances, tag_count):
    targets = {(i["dialogue_id"], i["target_turn_id"]): i for i in instances}
    out = ["# DialogueContext — data preview", "",
           "`scripts/author_dialogues.py` 가 만든다. 직접 고치지 않는다.", "",
           f"대화 {len(DIALOGUES)}편, 턴 {len(dialogues)}개, 인스턴스 {len(instances)}개.", "",
           "## 태그별 인스턴스 수", "", "| 태그 | 인스턴스 |", "|---|---|"]
    for tag in TAGS:
        ids = ", ".join(i["instance_id"] for i in instances if tag in i["challenge_tags"])
        out.append(f"| `{tag}` | {tag_count[tag]} — {ids} |")
    out.append("")
    for d in DIALOGUES:
        did = d["dialogue_id"]
        out += [f"## {did} — {d['setting']}", ""]
        out += [f"- **{k}**: {v}" for k, v in d["speakers"].items()]
        out += ["", "| # | 화자 | ko | en | 조각 | 목표 |", "|---|---|---|---|---|---|"]
        for r in dialogues:
            if r["dialogue_id"] != did:
                continue
            inst = targets.get((did, r["turn_id"]))
            mark = f"**{inst['instance_id']}**" if inst else ""
            frag = "→" if r["fragment_of_next"] else ""
            out.append(f"| {r['turn_id']} | {r['speaker']} | {md_cell(r['ko'])} | {md_cell(r['en'])} | {frag} | {mark} |")
        out.append("")
        for inst in instances:
            if inst["dialogue_id"] != did:
                continue
            out += [f"### {inst['instance_id']} — {', '.join('`'+t+'`' for t in inst['challenge_tags'])}", "",
                    f"- 현재 발화 ({inst['current_speaker']}): {inst['current_source']}",
                    f"- 정답: {inst['reference_translation']}"]
            out += [f"- check `{c['tag']}`: {c['requirement']}" for c in inst["checks"]]
            out += [f"- 문맥 효과: {inst['expected_context_effect']}", f"- 메모: {inst['note']}", ""]
    return "\n".join(out)


def main():
    dialogues, instances = build()
    tag_count = validate(dialogues, instances)
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "dialogues.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in dialogues:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / "instances.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in instances:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    (DATA / "preview.md").write_text(preview(dialogues, instances, tag_count) + "\n", encoding="utf-8")
    print(f"dialogues={len(DIALOGUES)} turns={len(dialogues)} instances={len(instances)}")
    for tag in TAGS:
        print(f"  {tag:22s} {tag_count[tag]}")


if __name__ == "__main__":
    main()
