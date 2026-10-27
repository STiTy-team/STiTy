import { canStart, nextPhase, type ConversationPhase } from './conversation-machine';

describe('conversation machine', () => {
  it('walks start → recording ⇄ paused → end', () => {
    let phase: ConversationPhase = 'idle';
    for (const [event, expected] of [
      ['start', 'starting'],
      ['started', 'recording'],
      ['pause', 'paused'],
      ['resume', 'resuming'],
      ['started', 'recording'],
      ['end', 'ending'],
      ['ended', 'idle'],
    ] as const) {
      phase = nextPhase(phase, event) ?? phase;
      expect(phase).toBe(expected);
    }
  });

  it('fails a start to failed and a resume back to paused', () => {
    expect(nextPhase('starting', 'startFailed')).toBe('failed');
    expect(nextPhase('resuming', 'resumeFailed')).toBe('paused');
    expect(canStart('failed')).toBe(true);
  });

  it('refuses events that do not apply', () => {
    expect(nextPhase('starting', 'start')).toBeNull();
    expect(nextPhase('idle', 'pause')).toBeNull();
    expect(nextPhase('paused', 'started')).toBeNull();
    expect(nextPhase('ending', 'start')).toBeNull();
  });

  it('pauses on the background from any running phase', () => {
    for (const phase of ['starting', 'resuming', 'recording'] as const) {
      expect(nextPhase(phase, 'background')).toBe('paused');
    }
  });
});
