#!/usr/bin/env python3
from pathlib import Path
import argparse, ast, json, re
import numpy as np


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--benchmark-dir', default='/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/05_paired_image_benchmark_120')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--out', default='~/methane_release_project/methanesat_methanefuse_native_compatibility_audit')
    return p.parse_args()


def find_repo(explicit):
    h=Path.home(); cwd=Path.cwd()
    cs=[Path(explicit).expanduser()] if explicit else []
    cs += [cwd, cwd/'MethaneFuse', h/'MethaneFuse', h/'methane_release_project'/'MethaneFuse', h/'methane_release_project']
    for c in cs:
        if (c/'scripts/eval/evaluate_classification.py').exists() and (c/'src/data/multisensor.py').exists() and (c/'src/models/finetune_loramoe_adapter.py').exists():
            return c
    raise SystemExit('MethaneFuse repo not found. Rerun with --repo /path/to/MethaneFuse')


def txt(p): return p.read_text(encoding='utf-8', errors='replace')


def main():
    a=args(); repo=find_repo(a.repo)
    model=txt(repo/'src/models/finetune_loramoe_adapter.py')
    data=txt(repo/'src/data/multisensor.py')
    trans=txt(repo/'src/data/sensor_transforms.py')
    ev=txt(repo/'scripts/eval/evaluate_classification.py')

    m=re.search(r'sensors\s*:\s*Sequence\[str\]\s*=\s*(\([^)]+\))', model)
    sensors=list(ast.literal_eval(m.group(1))) if m else []
    prefixes=sorted(set(re.findall(r'_wide_cols\("([^"]+)"\)', data)) | ({'s5p'} if 's5p_0_path' in data else set()))
    if "mapping emit_*_path to sensor key 'wv3'" in data: prefixes.append('emit->wv3')

    mentions=[]
    for p in repo.rglob('*.py'):
        try:
            if 'methanesat' in txt(p).lower(): mentions.append(str(p.relative_to(repo)))
        except Exception: pass

    s5p_repeat=('img.shape[0] == 1 and expected_channels > 1' in data and 'img.repeat(expected_channels' in data)
    s5p_norm='img = (img - self._s5p_mean) / self._s5p_std' in data
    sm=re.search(r'S5P_PRECOMPUTED_STATS\s*=\s*\(\s*(\[[^\]]*\])\s*,\s*(\[[^\]]*\])', trans, re.S)
    s5p_channels=None
    if sm:
        try: s5p_channels=len(ast.literal_eval(sm.group(1)))
        except Exception: pass

    bench=Path(a.benchmark_dir).expanduser(); pair_npz=next(iter(sorted((bench/'npz/pairs').glob('*.npz'))), None) if (bench/'npz/pairs').exists() else None
    bench_info={'canonical_manifest_exists':(bench/'manifests/06_repaired_primary_pairs.csv').exists(),'pair_npz_found':pair_npz is not None}
    if pair_npz:
        with np.load(pair_npz, allow_pickle=False) as z:
            bench_info['positive_xch4_shape']=list(z['positive_xch4'].shape)
            bench_info['keys']=list(z.files)

    ckpt=None
    if a.checkpoint:
        p=Path(a.checkpoint).expanduser(); ckpt=p if p.exists() else None
    else:
        for pat in ['checkpoints/**/*480m*.pt','checkpoints/**/*.pt','**/methanefuse_cls_480m.pt','**/stage2_classification_480m.pt']:
            ms=sorted(repo.glob(pat))
            if ms: ckpt=ms[0]; break

    ckinfo={'found':ckpt is not None,'path':str(ckpt) if ckpt else None}
    if ckpt:
        try:
            import torch
            obj=torch.load(ckpt,map_location='cpu')
            state=obj.get('model') or obj.get('state_dict') or obj.get('model_state_dict') or obj if isinstance(obj,dict) else {}
            keys=list(state.keys()) if isinstance(state,dict) else []
            hits={s:sum(bool(re.search(rf'(^|[._]){s}([._]|$)',k.lower())) for k in keys) for s in ['s2','l89','s5p','wv3','methanesat']}
            ckinfo.update({'inspected':True,'sensor_key_hits':hits,'key_count':len(keys)})
        except Exception as e:
            ckinfo.update({'inspected':False,'error':f'{type(e).__name__}: {e}'})

    native=('methanesat' in [s.lower() for s in sensors]) and bool(mentions)
    decision='NATIVE_METHANESAT_SUPPORT_FOUND' if native else 'NO_NATIVE_METHANESAT_SUPPORT'
    report={
        'repo':str(repo),'decision':decision,'model_default_sensors':sensors,'loader_prefixes':prefixes,
        'methanesat_python_mentions':mentions,'eval_uses_TriSensorTemporalCsvDataset':'TriSensorTemporalCsvDataset' in ev,
        'eval_uses_MultiSensorPanopticonClassifier':'MultiSensorPanopticonClassifier' in ev,
        's5p_single_channel_repeat':s5p_repeat,'s5p_normalization_applied':s5p_norm,'s5p_stats_channels':s5p_channels,
        'benchmark':bench_info,'checkpoint':ckinfo,
        'scientific_decision':'Do not masquerade MethaneSAT as S5P for claimed MethaneFuse inference.' if not native else 'Native path exists; verify exact preprocessing before inference.'
    }
    out=Path(a.out).expanduser(); out.mkdir(parents=True,exist_ok=True)
    (out/'compatibility_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    md=[
        '# MethaneSAT → MethaneFuse native compatibility audit','',f'**Decision: {decision}**','',
        f'- Model default sensors: `{sensors}`',f'- Loader prefixes: `{prefixes}`',f'- Python files mentioning MethaneSAT: `{mentions}`',
        f'- S5P repeats single-channel input to expected channels: `{s5p_repeat}`',f'- S5P normalization applied: `{s5p_norm}`',f'- S5P stats channel count: `{s5p_channels}`',
        f'- Example MethaneSAT pair shape: `{bench_info.get("positive_xch4_shape")}`','',
        '## Decision','',
        'Do **not** run the released MethaneFuse checkpoint on MethaneSAT by relabeling MethaneSAT as S5P.' if not native else 'Native MethaneSAT support exists; verify preprocessing and checkpoint parameters before inference.',
        '', 'The S5P loader may mechanically accept a single-channel NPZ and repeat channels, but that uses S5P-specific preprocessing/sensor identity and is not a MethaneSAT-native evaluation.'
    ]
    (out/'COMPATIBILITY_AUDIT.md').write_text('\n'.join(md)+'\n')
    print('='*80); print('METHANESAT -> METHANEFUSE NATIVE COMPATIBILITY AUDIT'); print('='*80)
    print('Repo:',repo); print('Model sensors:',sensors); print('Loader prefixes:',prefixes); print('MethaneSAT mentions:',mentions)
    print('S5P repeat 1ch:',s5p_repeat); print('S5P normalization:',s5p_norm); print('S5P stat channels:',s5p_channels)
    print('Benchmark example shape:',bench_info.get('positive_xch4_shape'))
    print('Checkpoint:',ckinfo); print(); print('DECISION:',decision)
    if not native: print('DO NOT masquerade MethaneSAT as S5P for a claimed MethaneFuse evaluation.')
    print('Saved:',out/'COMPATIBILITY_AUDIT.md'); print('Saved:',out/'compatibility_audit.json')

if __name__=='__main__': main()
