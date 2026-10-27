import { commonPrefixLength, typingInterval } from './typewriter';

describe('commonPrefixLength', () => {
  it('counts shared leading characters', () => {
    expect(commonPrefixLength('hello world', 'hello there')).toBe(6);
    expect(commonPrefixLength('', 'abc')).toBe(0);
    expect(commonPrefixLength('abc', 'abc')).toBe(3);
  });
});

describe('typingInterval', () => {
  it('stays within the 10–45 ms band', () => {
    expect(typingInterval(1)).toBe(45);
    expect(typingInterval(1_000)).toBe(10);
    expect(typingInterval(52)).toBe(25);
  });
});
