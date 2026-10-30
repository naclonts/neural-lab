"""Frozen exchangeable-set screen; execution exclusively in the queued CPU worker."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('OMP_NUM_THREADS', '4')
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint

REGIMES = ('mean', 'dispersion')
ARMS = ('pool', 'weighted_pool', 'linear_phi', 'raw_dense', 'sorted_dense', 'augmented_dense')
PARAMETERS = dict(zip(ARMS, (930, 930, 930, 5618, 5618, 5618)))
STEPS = 120


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value, name):
    if value is None or not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite or missing ' + name)


def canonical(x):
    second = torch.argsort(x[..., 1], dim=-1, stable=True)
    intermediate = x.gather(-2, second[..., None].expand(*second.shape, 2))
    first = torch.argsort(intermediate[..., 0], dim=-1, stable=True)
    indices = second.gather(-1, first)
    return x.gather(-2, indices[..., None].expand(*indices.shape, 2)), indices


def permute(x, indices):
    return x.gather(1, indices[..., None].expand(*indices.shape, 2))


def task(regime, r, j):
    teacher_seed = 51100 + 100*r + j
    rng = torch.Generator().manual_seed(teacher_seed)
    rejected_a = 0
    while True:
        a = torch.randn((2,), generator=rng, dtype=torch.float32)
        if a.norm().item() >= 1e-6:
            a = a / a.norm()
            break
        rejected_a += 1
    b = .06*torch.rand((), generator=rng, dtype=torch.float32) - .03
    arrays = {'teacher_a': a, 'teacher_b': b}
    seen, rejected_sets = set(), {}
    split_seeds = {'train': 52100+100*r+10*j, 'dev': 52101+100*r+10*j}
    for split, size in (('train', 256), ('dev', 512)):
        generator = torch.Generator().manual_seed(split_seeds[split])
        rows, rejected = [], 0
        while len(rows) < size:
            x = 2*torch.rand((8, 2), generator=generator, dtype=torch.float32)-1
            ordered, _ = canonical(x)
            identity = tuple(ordered.flatten().tolist())
            if identity in seen:
                rejected += 1
                continue
            seen.add(identity)
            rows.append(x)
        x = torch.stack(rows)
        ordered, indices = canonical(x)
        z = x @ a
        mean_score = z.mean(-1)-b
        dispersion_score = z.square().mean(-1)-1/3-b
        score = mean_score if regime == 'mean' else dispersion_score
        arrays.update({split+'_inputs': x, split+'_sorted_inputs': ordered,
                       split+'_sort_indices': indices, split+'_z': z,
                       split+'_mean_score': mean_score, split+'_dispersion_score': dispersion_score,
                       split+'_teacher_score': score, split+'_labels': (score > 0).long()})
        rejected_sets[split] = rejected
    extra_seed, augmentation_seed = 53100+100*r+j, 54100+100*r+j
    generator = torch.Generator().manual_seed(extra_seed)
    indices = torch.stack([torch.randperm(8, generator=generator) for _ in range(512)])
    arrays['dev_permutation_indices'] = indices
    arrays['dev_permuted_inputs'] = permute(arrays['dev_inputs'], indices)
    generator = torch.Generator().manual_seed(augmentation_seed)
    arrays['train_augmentation_indices'] = torch.stack([
        torch.stack([torch.randperm(8, generator=generator) for _ in range(256)])
        for _ in range(STEPS)])
    assert len(seen) == 768
    return arrays, {'teacher_seed': teacher_seed, 'split_seeds': split_seeds,
                    'extra_dev_permutation_seed': extra_seed, 'augmentation_seed': augmentation_seed,
                    'a_norm_rejections': rejected_a, 'duplicate_set_rejections': rejected_sets}


class Pool(nn.Module):
    def __init__(self, arm='pool'):
        super().__init__()
        self.arm = arm
        self.phi = nn.ModuleList([nn.Linear(2, 16), nn.Linear(16, 16)])
        self.rho = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x):
        h = x
        for layer in self.phi:
            h = layer(h)
            if self.arm != 'linear_phi':
                h = F.gelu(h)
        if self.arm == 'weighted_pool':
            weights = 2*torch.arange(1, 9, dtype=h.dtype, device=h.device)/9
            h = h*weights[None, :, None]
        return self.rho(h.mean(1))


class Dense(nn.Module):
    def __init__(self, arm='raw_dense'):
        super().__init__()
        self.arm = arm
        self.input = nn.Linear(16, 48)
        self.blocks = nn.ModuleList([nn.Linear(48, 48) for _ in range(2)])
        self.output = nn.Linear(48, 2)

    def forward(self, x):
        if self.arm == 'sorted_dense':
            x, _ = canonical(x)
        h = F.gelu(self.input(x.flatten(1)))
        for block in self.blocks:
            h = h + F.gelu(block(h))
        return self.output(h)


def activation_check(module, inputs, output):
    if isinstance(output, torch.Tensor):
        finite(output, type(module).__name__+' activation')


@torch.no_grad()
def evaluate(model, arrays, known_train_loss):
    model.eval()
    result, logits_by_split = {}, {}
    for split in ('train', 'dev'):
        logits = model(arrays[split+'_inputs'])
        loss = logits.new_tensor(known_train_loss) if split == 'train' else F.cross_entropy(logits, arrays[split+'_labels'])
        finite(logits, split+' logits')
        finite(loss, split+' CE')
        logits_by_split[split] = logits
        result[split] = {'loss': loss.item(),
                         'accuracy': (logits.argmax(-1) == arrays[split+'_labels']).float().mean().item(),
                         'logit_rms': logits.square().mean().sqrt().item()}
    other = model(arrays['dev_permuted_inputs'])
    finite(other, 'permuted dev logits')
    difference = (other-logits_by_split['dev']).abs()
    other_loss = F.cross_entropy(other, arrays['dev_labels'])
    finite(other_loss, 'permuted dev CE')
    result['permutation'] = {'mean_absolute_logit_difference': difference.mean().item(),
                             'max_absolute_logit_difference': difference.max().item(),
                             'prediction_disagreement': (other.argmax(-1) != logits_by_split['dev'].argmax(-1)).float().mean().item(),
                             'dev_loss': other_loss.item(), 'dev_loss_change': other_loss.item()-result['dev']['loss']}
    result['dev_minus_train_loss'] = result['dev']['loss']-result['train']['loss']
    return result


def config(arm, source_sha):
    return {'script': 'permutation_pool_classifier.py', 'source_sha256': source_sha, 'arm': arm,
            'architecture': 'phi Linear2,16 GELU Linear16,16 GELU; mean; rho Linear16,32 GELU Linear32,2; linear_phi omits phi GELUs; weighted_pool multiplies item i by2*(i+1)/9 before mean; dense Linear16,48 GELU then2 residual Linear48,48 GELU then Linear48,2',
            'preprocessing': 'inputs Nx8x2; sorted_dense stable second-coordinate then stable first-coordinate lexicographic sort and flatten; raw/augmented dense flatten; no normalization',
            'trainable_parameters': PARAMETERS[arm], 'optimizer': 'AdamW', 'learning_rate': .003,
            'weight_decay': .001, 'betas': [.9, .999], 'epsilon': 1e-8, 'steps': STEPS,
            'batch': 'full256', 'device': 'cpu', 'threads': 4,
            'selection': 'all observed training CE at actual current weights, earliest strict minimum; augmented canonical121 plus scheduled120 views; others canonical121',
            'endpoint': 'independent final120 original-order dev; never reload selected weights'}


def fit(model, arrays, writer, arm, fit_id, dataset, source_sha, init_index):
    started = time.monotonic()
    hooks = [module.register_forward_hook(activation_check) for module in model.modules()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001, betas=(.9, .999), eps=1e-8)
    best = BestCheckpoint(scope='minimum all observed training CE at actual states; earliest strict minimum; augmented original and scheduled views; no held-out selection')
    observations, canonical_trace = [], []

    def observe(loss, step, view):
        finite(loss, view+' training CE')
        value = loss.item()
        best.observe(model, value, step)
        observations.append({'step': step, 'view': view, 'value': value})
        return value

    model.eval()
    loss = F.cross_entropy(model(arrays['train_inputs']), arrays['train_labels'])
    canonical_trace.append(observe(loss, 0, 'original'))
    initial = evaluate(model, arrays, canonical_trace[0])
    for step in range(1, STEPS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if arm == 'augmented_dense':
            inputs = permute(arrays['train_inputs'], arrays['train_augmentation_indices'][step-1])
            gradient_loss = F.cross_entropy(model(inputs), arrays['train_labels'])
            observe(gradient_loss, step-1, 'augmentation_schedule_'+str(step-1))
        else:
            gradient_loss = loss
        gradient_loss.backward()
        for name, parameter in model.named_parameters():
            finite(parameter.grad, 'gradient '+name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            finite(parameter, 'parameter '+name)
        model.eval()
        loss = F.cross_entropy(model(arrays['train_inputs']), arrays['train_labels'])
        canonical_trace.append(observe(loss, step, 'original'))
    final = evaluate(model, arrays, canonical_trace[-1])
    if arm in ('pool', 'linear_phi', 'sorted_dense'):
        assert final['permutation']['max_absolute_logit_difference'] <= 1e-5
    expected = 241 if arm == 'augmented_dense' else 121
    assert best.observations == len(observations) == expected and len(canonical_trace) == 121
    configuration = {**config(arm, source_sha), 'fit_id': fit_id,
                     'pool_init_seed': 55101+init_index, 'dense_init_seed': 56101+init_index,
                     'dataset_path': dataset['path'], 'dataset_sha256': dataset['sha256']}
    checkpoint = best.save(writer, fit_id+'-'+arm+'-best', arm=arm, seed=(55101 if arm in ARMS[:3] else 56101)+init_index,
                           config=configuration, datasets=[dataset['path']])
    minimum = min(row['value'] for row in observations)
    earliest = next(row for row in observations if row['value'] == minimum)
    assert checkpoint['selection']['value'] == minimum and checkpoint['selection']['step'] == earliest['step']
    for hook in hooks:
        hook.remove()
    return {'config': configuration, 'canonical_training_loss_every_state': canonical_trace,
            'selection_observations': observations, 'initial_metrics': initial,
            'final_update': STEPS, 'final_update_metrics': final,
            'best_training_checkpoint': checkpoint['path'], 'best_training_selection': checkpoint['selection'],
            'selected_observation_view': earliest['view'], 'elapsed_seconds': time.monotonic()-started,
            'activations_gradients_parameters_finite': True}


def aggregate(records):
    metrics, comparisons, summaries = {}, {}, {}
    controls = {'mean': ('raw_dense',), 'dispersion': ARMS[1:]}
    for regime in REGIMES:
        rows = [row for row in records if row['regime'] == regime]
        summaries[regime], comparisons[regime] = {}, {}
        for arm in ARMS:
            endpoints = {}
            for endpoint in ('loss', 'accuracy', 'dev_minus_train_loss'):
                teachers = []
                for j in range(3):
                    paired = [row for row in rows if row['teacher_index'] == j]
                    values = [row['arms'][arm]['final_update_metrics']['dev'][endpoint] if endpoint != 'dev_minus_train_loss'
                              else row['arms'][arm]['final_update_metrics'][endpoint] for row in paired]
                    teachers.append({'teacher_index': j, 'init_values': values, 'mean': sum(values)/2})
                means = [row['mean'] for row in teachers]
                endpoints[endpoint] = {'teachers': teachers, 'regime_mean': sum(means)/3,
                                       'teacher_min': min(means), 'teacher_max': max(means)}
            summaries[regime][arm] = endpoints
            for endpoint in ('loss', 'accuracy'):
                metrics[f'{regime}_{arm}_dev_{endpoint}'] = endpoints[endpoint]['regime_mean']
        for control in controls[regime]:
            teachers = []
            for j in range(3):
                paired = [row for row in rows if row['teacher_index'] == j]
                differences = [row['arms']['pool']['final_update_metrics']['dev']['loss']-row['arms'][control]['final_update_metrics']['dev']['loss'] for row in paired]
                teachers.append({'teacher_index': j, 'init_differences': differences, 'mean': sum(differences)/2})
            means = [row['mean'] for row in teachers]
            comparisons[regime][control] = {'teachers': teachers, 'mean': sum(means)/3, 'min': min(means), 'max': max(means)}
            metrics[f'{regime}_pool_minus_{control}_dev_loss'] = sum(means)/3
    dispersion = comparisons['dispersion']
    passed = (all(dispersion[arm]['mean'] <= -.02 and dispersion[arm]['max'] < 0 for arm in ('weighted_pool', 'linear_phi', 'raw_dense'))
              and all(dispersion[arm]['mean'] <= .01 for arm in ('sorted_dense', 'augmented_dense'))
              and comparisons['mean']['raw_dense']['mean'] <= .02)
    return metrics, comparisons, summaries, passed


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    writer = ArtifactWriter('/output')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared = []
    for r, regime in enumerate(REGIMES):
        for j in range(3):
            arrays, generation = task(regime, r, j)
            for name, value in arrays.items():
                finite(value, name)
            hashes = {name: digest(value) for name, value in arrays.items()}
            provenance = {'kind': 'generated analytic synthetic', 'script_sha256': source_sha,
                          'generator': 'permutation_pool_classifier.py:task', 'regime': regime, 'teacher_index': j,
                          **generation, 'pool_init_seeds': [55101, 55102], 'dense_init_seeds': [56101, 56102],
                          'draw_order': 'CPU float32; a randn2 reject norm<1e-6 then normalize; b=.06*rand()-.03; one8x2rand call/set train then dev; reject canonical-exact duplicates across splits; extra dev randperm8 example-major; augmentation randperm8 update-major then example-major; separate generators',
                          'teacher': 'z=X@a; mean=mean(z)-b; dispersion=mean(z^2)-1/3-b; label=score>0',
                          'preprocessing': 'canonical stable second coordinate then stable first coordinate; saved sort/permutation indices; augmentation schedule120x256x8 shared across initialization seeds',
                          'split': {'train': 256, 'dev': 512, 'test': 'not generated'},
                          'array_sha256': hashes,
                          'label_counts': {sp: arrays[sp+'_labels'].bincount(minlength=2).tolist() for sp in ('train', 'dev')},
                          'downloads': False, 'generator_software': {'torch': str(torch.__version__), 'device': 'cpu', 'dtype': 'float32; indices/labels int64'}}
            dataset = writer.save_dataset(regime+'-teacher-'+str(j), arrays, provenance=provenance)
            prepared.append((r, regime, j, arrays, generation, hashes, dataset))
    assert len(writer.manifest['datasets']) == 6 and not writer.manifest['checkpoints']
    records = []
    for r, regime, j, arrays, generation, hashes, dataset in prepared:
        for s in range(2):
            torch.manual_seed(55101+s)
            pool_reference = Pool()
            torch.manual_seed(56101+s)
            dense_reference = Dense()
            models = {arm: copy.deepcopy(pool_reference if arm in ARMS[:3] else dense_reference) for arm in ARMS}
            for arm, model in models.items():
                model.arm = arm
            assert {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()} == PARAMETERS
            initial_hashes = {arm: {name: digest(value) for name, value in model.state_dict().items()} for arm, model in models.items()}
            assert initial_hashes['pool'] == initial_hashes['weighted_pool'] == initial_hashes['linear_phi']
            assert initial_hashes['raw_dense'] == initial_hashes['sorted_dense'] == initial_hashes['augmented_dense']
            order = ARMS if ((r*3+j)*2+s)%2 == 0 else tuple(reversed(ARMS))
            fit_id = f'{regime}-teacher-{j}-init-{s}'
            row = {'regime': regime, 'teacher_index': j, 'init_index': s, 'pool_init_seed': 55101+s,
                   'dense_init_seed': 56101+s, 'generation': generation, 'dataset': dataset['path'],
                   'dataset_sha256': dataset['sha256'], 'array_sha256': hashes,
                   'initial_state_sha256': initial_hashes, 'trainable_parameters': PARAMETERS,
                   'arm_order': order, 'arms': {}}
            for arm in order:
                row['arms'][arm] = fit(models[arm], arrays, writer, arm, fit_id, dataset, source_sha, s)
            records.append(row)
    assert len(records) == 12 and len(writer.manifest['checkpoints']) == 72
    metrics, comparisons, summaries, passed = aggregate(records)
    details = {'status': 'completed', 'script_sha256': source_sha, 'elapsed_seconds': time.monotonic()-started,
               'config': {'regimes': REGIMES, 'teacher_indices': [0, 1, 2], 'pool_init_seeds': [55101, 55102],
                          'dense_init_seeds': [56101, 56102], 'arms': {arm: config(arm, source_sha) for arm in ARMS},
                          'primary_endpoint': 'independent final120 original-order dev CE; average2inits then equal3teachers',
                          'screening_rule': 'dispersion pool-minus weighted/linear/raw mean<=-.02 and each teacher<0; sorted/augmented mean<=.01; mean raw mean<=.02',
                          'stopping_rule': 'exactly120 updates; no dev selection, tuning, test or rescue'},
               'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu',
                               'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
               'records': records, 'per_arm_final_dev_summary': summaries, 'paired_final_dev_contrasts': comparisons,
               'screening_criterion_met': passed,
               'limitations': 'Established Deep Sets, deliberately symmetry-aligned synthetic development screen; three teacher clusters/regime and two initializations. No novelty, significance, equivalence, broad superiority, universality or causal identification. Weighted pooling changes variance/gradients; linear phi lowers functional capacity; dense budgets5618 versus930; equal updates do not match FLOPs/tuning/optimization. Augmentation changes trajectories and selector has241 mixed-view observations versus121 original-view observations; selected checkpoint quality is not comparable. Dev permutations are paired views, not independent replicates. No test/promotion; failed clauses stop without rescue.'}
    summary = 'Frozen DeepSets screen:2regimes x3teachers x2initializations x6arms=72fits,120CPUupdates. Six exact datasets and72all-observed training-CE checkpoints; augmented241observations,others121. Independent final120dev,paired permutation diagnostics; no test,tuning,promotion.'
    assert 1 <= len(metrics) <= 32 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values())
    assert len(summary) <= 2000
    output = Path('/output')
    (output/'details.json').write_text(json.dumps(details, allow_nan=False, indent=2))
    result = json.dumps({'metrics': metrics, 'summary': summary}, allow_nan=False, indent=2)
    (output/'result.json').write_text(result)
    print(result)


if __name__ == '__main__':
    main()
