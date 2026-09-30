# T14 Doc-COMET vs COMET / XCOMET by condition (models pooled)

Doc-COMET = Unbabel/wmt22-comet-da with the previous w gold turns (source and reference) prepended, pooled over the current turn only; w in [2, 5]. The scoring context is identical for every condition. *_gain = minus NONE.

| condition | n_rows | comet | doc_comet_w2 | doc_comet_w5 | xcomet | comet_gain | doc_comet_w2_gain | doc_comet_w5_gain | xcomet_gain |
|---|---|---|---|---|---|---|---|---|---|
| NONE | 80 | 0.797 | 0.716 | 0.719 | 0.852 | 0.000 | 0.000 | 0.000 | 0.000 |
| SRC@1 | 80 | 0.809 | 0.738 | 0.741 | 0.873 | 0.012 | 0.022 | 0.022 | 0.021 |
| SRC@3 | 80 | 0.823 | 0.755 | 0.759 | 0.895 | 0.026 | 0.039 | 0.041 | 0.043 |
| SRC@5 | 80 | 0.824 | 0.754 | 0.761 | 0.899 | 0.027 | 0.038 | 0.043 | 0.048 |
| TGT@1 | 80 | 0.806 | 0.737 | 0.742 | 0.874 | 0.009 | 0.021 | 0.024 | 0.022 |
| TGT@3 | 80 | 0.815 | 0.748 | 0.753 | 0.887 | 0.018 | 0.032 | 0.035 | 0.036 |
| TGT@5 | 80 | 0.828 | 0.759 | 0.761 | 0.911 | 0.031 | 0.043 | 0.043 | 0.059 |
| SRC_TGT@1 | 80 | 0.814 | 0.745 | 0.746 | 0.880 | 0.017 | 0.029 | 0.028 | 0.028 |
| SRC_TGT@3 | 80 | 0.827 | 0.767 | 0.768 | 0.894 | 0.030 | 0.051 | 0.050 | 0.042 |
| SRC_TGT@5 | 80 | 0.841 | 0.778 | 0.781 | 0.911 | 0.044 | 0.062 | 0.062 | 0.059 |
| SPK_SRC_TGT@1 | 80 | 0.812 | 0.744 | 0.749 | 0.874 | 0.015 | 0.028 | 0.031 | 0.022 |
| SPK_SRC_TGT@3 | 80 | 0.825 | 0.761 | 0.764 | 0.891 | 0.029 | 0.045 | 0.046 | 0.039 |
| SPK_SRC_TGT@5 | 80 | 0.833 | 0.767 | 0.771 | 0.905 | 0.036 | 0.051 | 0.053 | 0.053 |
