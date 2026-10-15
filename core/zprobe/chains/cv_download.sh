#!/bin/bash
# One CoVoST2 en->ja train shard (~36k English Common Voice clips with gold English text) for distillation audio.
cd /home/mobility/STiTy
.venv/bin/python -u -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('fixie-ai/covost2', 'en_ja/train-00000-of-00008.parquet', repo_type='dataset', local_dir='/home/mobility/datasets/covost2')
print('DOWNLOADED', p)
"
