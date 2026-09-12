"""Scoring for speech translation: WER, CER, BLEU, FSL, LAAL, commit mix, routing.

Two entry points. `score_item` scores one utterance as it lands, `score_run` scores
the corpus at the end. Both answer the same question -- what can this data support --
and neither ever raises for missing data.

**Everything available is computed and nothing is selected up front.** A subset
chosen in advance only hides numbers the GPU time was already spent on. What the
data cannot support is simply absent from the result and named in `unavailable`
with a reason, so the two failure modes stay distinguishable:

  a dataset of audio with no transcripts   -> wer, cer absent; fsl, commit present
  transcripts but no reference translation -> bleu, laal absent; wer, cer present
  references in only some languages        -> those targets absent from the
                                              breakdown, named in the reason

There are no nulls. A metric is either a number or a reason, never a bare `null`
that a reader has to guess about -- which is what makes "every missing metric has an
explanation" true by construction rather than by convention.

Nothing here knows about a benchmark run, a config file or a report format. The
caller supplies a `TargetPolicy` (one method, `expected_target`) and decides what to
do with what comes back.
"""
from dataclasses import dataclass, field

from . import asr, bleu, commit, latency, routing, text
from .langs_bridge import laal_unit
from .types import Segment, TargetPolicy, Utterance

__all__ = ["asr", "bleu", "commit", "latency", "routing", "text",
           "Segment", "Utterance", "TargetPolicy",
           "RunScore", "score_item", "score_run"]



@dataclass(frozen=True)
class RunScore:
    """What the run scored, what it could not, and which items misrouted.

    `aggregate` holds only computed numbers. `unavailable` maps a metric to why it
    is missing. `misrouted_items` is a list of ids for inspection, not a score.
    """

    aggregate: dict = field(default_factory=dict)
    unavailable: dict = field(default_factory=dict)
    misrouted_items: list = field(default_factory=list)

    def __contains__(self, key: str) -> bool:
        return key in self.aggregate

    def get(self, key: str, default=None):
        return self.aggregate.get(key, default)


def score_item(row, *, languages: TargetPolicy) -> dict:
    """Everything one utterance can be scored on, as a flat dict of fields.

    Takes a row mapping or an `Utterance`. Keys the data cannot support are absent
    rather than null, so a caller sorting or printing them tests for presence.
    """
    item = row if isinstance(row, Utterance) else Utterance.from_row(row)
    target = languages.expected_target(item.src_lang)

    fields = {"n_segments": len(item.segments),
              "route_errors": routing.route_errors(item, target_lang=target)}
    fields.update({k: v for k, v in asr.per_item(item).items() if v is not None})
    fsl = latency.fsl_stats(item.segments)
    if fsl["n_seg_with_fsl"]:
        fields.update(fsl)
    fields.update(latency.laal_for_item(item, target_lang=target,
                                       unit=laal_unit(target)))
    sentence_bleu = bleu.sentence(item.hypothesis_translation,
                                 item.reference_translation(target),
                                 target_lang=target)
    if sentence_bleu is not None:
        fields["sentence_bleu"] = sentence_bleu
    return fields


def score_run(rows, *, languages: TargetPolicy) -> RunScore:
    """Score the corpus. Rows may be mappings or `Utterance` objects.

    Only utterances that produced output are scored -- a caller's failed rows carry
    no hypothesis and would otherwise be counted as perfect deletions of nothing.
    Pass only the rows worth scoring.
    """
    items = [r if isinstance(r, Utterance) else Utterance.from_row(r) for r in rows]

    aggregate: dict = {}
    unavailable: dict = {}
    for values, missing in (
        asr.corpus(items),
        latency.corpus(items, languages=languages, unit_for=laal_unit),
        bleu.corpus(items, languages=languages),
        commit.corpus(items),
        routing.corpus(items, languages=languages),
    ):
        aggregate.update(values)
        unavailable.update(missing)

    return RunScore(aggregate=aggregate, unavailable=unavailable,
                    misrouted_items=aggregate.pop("misrouted_items", []))
