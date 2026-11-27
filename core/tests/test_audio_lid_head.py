import torch

from core.components.langid.audio.head import LidHead


def test_padding_does_not_change_a_short_clip():
    torch.manual_seed(0)
    head = LidHead(3).eval()
    short = torch.randn(1, 5, 768)
    alone = head(short, torch.ones(1, 5, dtype=torch.bool))
    padded = torch.cat([short, torch.randn(1, 7, 768)], dim=1)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, :5] = True
    assert torch.allclose(alone, head(padded, mask), atol=1e-5)


def test_save_load_round_trip(tmp_path):
    head = LidHead(2).eval()
    x, m = torch.randn(2, 4, 768), torch.ones(2, 4, dtype=torch.bool)
    path = tmp_path / "h.pt"
    head.save(path, ["ko", "en"], {"note": "t"})
    again, langs, meta = LidHead.load(path)
    assert langs == ["ko", "en"] and meta["note"] == "t"
    assert torch.allclose(head(x, m), again.eval()(x, m))
