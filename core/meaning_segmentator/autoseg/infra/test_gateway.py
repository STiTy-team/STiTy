"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.infra.test_gateway`"""
import unittest

from . import gateway as G


class KeyRotation(unittest.TestCase):
    """키 여러 개(다른 조직)를 호출마다 돌려 쓴다 — 조직별 처리 큐를 둘로."""

    def test_round_robin(self):
        gw = G.Gateway(provider="openai", api_key="k0", api_keys=["k0", "k1"], model="gpt-5-mini")
        self.assertEqual([gw._next_key()[0] for _ in range(5)], [0, 1, 0, 1, 0])
        self.assertEqual(gw._next_key()[1], "k1")

    def test_single_key_default(self):
        gw = G.Gateway(provider="openai", api_key="k0", model="gpt-5-mini")
        self.assertEqual([gw._next_key() for _ in range(3)], [(0, "k0")] * 3)

    def test_connection_limit_follows_workers(self):
        gw = G.Gateway(provider="openai", api_key="k0", model="gpt-5-mini", max_connections=64)
        self.assertEqual(gw._client._transport._pool._max_connections, 64)


class UsageByKey(unittest.TestCase):
    def test_by_key_accounting(self):
        u = G.Usage(price=(1.0, 0.5, 2.0))
        payload = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}, "choices": [{}]}
        u.add(payload, "segment", key=0)
        u.add(payload, "segment", key=1)
        u.add(payload, "segment", key=1)
        snap = u.snapshot()
        self.assertEqual(snap["by_key"]["0"]["calls"], 1)
        self.assertEqual(snap["by_key"]["1"]["calls"], 2)
        self.assertAlmostEqual(snap["by_key"]["1"]["cost"], 2.0)
        self.assertAlmostEqual(snap["cost"], 3.0)
