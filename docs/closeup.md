# Close-up: a toolbox for looking hard at a few seconds

`voicememowhisper/si/closeup.py`

The main pipeline makes one pass over an hour of audio and has to be fast
enough that you will actually run it. Close-up makes the opposite trade: it is
allowed to be slow, wasteful, and repetitive, because you only ever point it at
a few seconds — and on a few seconds, "slow" means a handful of seconds.

That trade buys things a single pass cannot get. A 0.2-second "嗯" that hides
inside a speaker's word boundaries. A muttered sentence the model turned to
mush. Whether the person you think is talking is really the one talking.

**Nothing in `si run` calls this, and nothing should.** The moment it runs over
a whole recording the cost model collapses. It is a magnifying glass: a human
picks up the glass and points it somewhere.

## The shape of the thing

One idea underneath everything here:

> **Run the same few seconds through several different treatments, and compare
> what comes back.**

Where the treatments agree, you have something solid. Where they disagree, you
have found the hard part — which is itself the useful signal, because it tells
you what not to trust.

A **variant** is one treatment: a chain of audio transforms plus a recognizer.
An analysis is a set of variants, run over the same window, with their outputs
lined up against each other.

```
                    ┌── denoise → 0.25x → whisper-medium ──┐
window [t0, t1) ────┼── gain +10dB → 0.5x → whisper-large ─┼──→ compare → report
                    └── raw → 1.0x → funasr ───────────────┘
```

The comparison is where the value is. The transforms are just plumbing.

## Audio transforms

All of these are single ffmpeg filters, which is why combining them costs
nothing: they concatenate into one filter chain and one pass over the audio.
`audio_filter()` builds the chain; `apply_filters()` runs it.

| Transform | Filter | What it is for |
|---|---|---|
| **Slowdown** | `atempo` cascade | The workhorse. At 1.0x a 0.2s token has too few frames for the model to emit it separately, and it gets swallowed by its neighbours. At 0.25x it has 4x the frames and comes out as its own segment. Below 0.5x needs a cascade — `atempo` bottoms out at 0.5. |
| **Denoise** | `afftdn` | Spectral noise reduction. Hiss, fan, air conditioning, room tone. Helps most when the thing you are chasing is quiet and the noise floor is what is burying it. Can hurt: it eats the breathy onset of quiet speech too. |
| **Gain** | `volume` | Brute amplification. For a talker who was far from the mic. Clips if you overdo it. |
| **Loudness normalize** | `loudnorm` | Gain, but aiming at a target loudness rather than a fixed boost. Safer than `volume` when you do not know how hot the source is. |
| **High-pass** | `highpass` | Cuts rumble below speech (mains hum, HVAC, handling noise). Cheap and rarely harmful — speech does not live below ~80 Hz. |
| **Low-pass** | `lowpass` | Cuts hiss above speech. Blunter, use with care. |

The catch worth knowing: **these interact, and not always in your favour.**
Denoising then amplifying is not the same as amplifying then denoising.
Aggressive `afftdn` can erase exactly the faint acknowledgement you were
hunting. There is no universally correct chain — that is precisely why the tool
runs several and compares, instead of picking one and trusting it.

## Recognizers

Anything with the shape `(wav_path) -> [(start, end, text), ...]`. Today the
CLI wires up faster-whisper; the interface deliberately does not care.

Reasons to run more than one:

- **Same model, several passes.** Not pointless: it exposes where the model is
  unstable. A segment three runs agree on is different in kind from one that
  appears in a single run.
- **Same model, different speeds.** The 0.25x pass and the 1.0x pass fail
  differently. A token that survives both is real.
- **Different models.** Genuinely independent errors. The most informative
  disagreement you can get, and the most expensive.

## Comparisons

What you do with the outputs. This is the part with the most room to grow.

**Implemented:**

- **Vote.** Segments that several variants heard at the same moment fold into
  one, carrying a count. `votes == 1` means one run out of N imagined it, or
  saw something the others missed — either way, look closer.
- **Acknowledgement classifier.** Text that looks like a listener's "嗯 / 对 /
  好的" rather than someone taking the floor. Deliberately shallow, and
  deliberately generous: on this material a false positive costs you a second
  of clip, a false negative costs you a poisoned speaker centroid.
- **Clean subranges.** The stretches of a window with no acknowledgement in
  them — which is the answer to "where in this block can I safely cut an
  enrollment clip from".

**Not yet, in rough order of how much they would buy:**

- **Voiceprint comparison per segment.** Embed each candidate segment and cosine
  it against the speaker library. This turns the acknowledgement classifier from
  a guess into a measurement: right now "嗯" is flagged because of *what was
  said*, and a speaker's own "嗯" gets flagged along with the listener's. The
  embedding knows *who said it*. This is the single biggest upgrade available,
  and it is the thing that would let vetting run unattended.
- **Text-based speaker discrimination.** Hand the transcript to an LLM and ask
  who is speaking, from the content alone: turn-taking, who asks and who
  answers, who owns which jargon, who refers to whom in the third person. Works
  exactly where acoustics fail — the two people whose voices are hard to tell
  apart usually do not say interchangeable things. A judgement made on a
  different axis than the embedding, so its agreement means something.
- **Cross-variant text diff.** Do not just vote on identical strings, diff them.
  Where three variants produce three different words for one moment, the
  disagreement localises the hard part to a single word, and the alternatives
  are themselves the candidate list.
- **Speed sweep.** Try 0.5x / 0.35x / 0.25x / 0.125x and keep what is stable
  across them. The optimum is recording-dependent and there is no reason to
  guess it when trying is this cheap.
- **Filter-chain sweep.** Same, over denoise/gain/high-pass combinations. Report
  which chain made a stubborn passage legible, so the next recording from that
  mic starts from a better default.
- **Forced alignment.** Align the transcript to the audio to get real word
  boundaries, instead of inferring them from segment timestamps. Makes the
  0.15–0.5s duration gate a lot less crude.
- **Overlap detection.** Two people talking at once is the case every part of
  this pipeline handles worst, and it is exactly when a backchannel happens.
  Detecting overlap directly would beat inferring it from an energy dip.

## Energy gate

There is also a purely acoustic path — no ASR at all — that finds candidates by
energy alone: runs that are audible but markedly weaker than the person holding
the floor, and short. It exists because a backchannel is sometimes too quiet or
too slurred for any ASR to emit, at any speed, and it still needs to be found
before it lands in an enrollment clip.

Its thresholds are **derived from the window itself** (`adaptive_thresholds`):
the noise floor and the speech level of *this* window, with the gate placed
between them. An earlier version used absolute dB tuned on one reference
recording; those numbers drifted the instant the mic moved. What defines a
backchannel is a relative fact, so it is expressed as one.

## Adding to the toolbox

The three layers are independent on purpose:

- A **transform** is a filter string. Add it to `audio_filter()`.
- A **recognizer** is `(path) -> [(start, end, text)]`. Anything satisfying that
  works.
- A **comparison** reads a `WindowReport` and reports something. It does not
  need to know how the audio got there.

If a new idea does not fit in one of those three, that is worth a second
thought before it goes in — it may be a pipeline stage wearing a disguise, and
pipeline stages belong in `si run`, where they get the cost scrutiny they
deserve.

## Testing

The RMS core is pure functions over dB arrays (`adaptive_thresholds`,
`segments_from_db`), and the recognizer is an injected callable. So the tests
need neither ffmpeg nor a real recording, and they run in milliseconds. Keep it
that way: anything that can only be tested against a real audio file will not
be tested.

## Cost

The honest numbers, on a ten-second window: cutting and filtering is
milliseconds. A slowed pass is 4x the audio, so a 0.25x pass over 10s is 40s of
audio to transcribe. Three variants of that is a couple of minutes on CPU. That
is fine for something a human asked for and is waiting on, and it is completely
unaffordable over an hour of tape — 3 variants × 4x × 1h is half a day. The
whole design follows from that ratio.
