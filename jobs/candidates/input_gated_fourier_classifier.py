"""Frozen input-conditioned raw/Fourier logit mixture; queued CPU worker only."""
import copy
import hashlib
import json
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

REGIMES = ('affine', 'periodic', 'mixed')
ARMS = ('input_gate', 'constant_gate', 'static_blend', 'concat', 'fourier', 'raw_dense')
INIT_SEEDS = (41101, 41102)
PARAMETERS = dict(zip(ARMS, (1131, 1131, 1128, 1186, 1122, 4946)))
STEPS, SIZE = 120, 512
DIAGNOSTIC_STEPS = (0, 40, 80, 120)


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite ' + name)


def task(regime, r, j):
    teacher_seed = 39100 + 100*r + j
    rng = torch.Generator().manual_seed(teacher_seed)
    a_rejections = 0
    while True:
        a = torch.randn((2,), generator=rng, dtype=torch.float32)
        if a.norm().item() >= 1e-6:
            a = a / a.norm()
            break
        a_rejections += 1
    b = .3 * torch.rand((), generator=rng, dtype=torch.float32) - .15
    frequencies, q_rejections = [], 0
    while len(frequencies) < 2:
        q = torch.randn((2,), generator=rng, dtype=torch.float32) * 1.5
        if .75 <= q.norm().item() <= 3:
            frequencies.append(q)
        else:
            q_rejections += 1
    q = torch.stack(frequencies)
    phi = 2 * torch.pi * torch.rand((2,), generator=rng, dtype=torch.float32)
    arrays = {'teacher_a': a, 'teacher_b': b, 'teacher_q': q, 'teacher_phi': phi}
    seeds = (40100 + 100*r + 10*j, 40101 + 100*r + 10*j)
    rejections, seen = {}, set()
    for split, seed in zip(('train', 'dev'), seeds):
        generator = torch.Generator().manual_seed(seed)
        rows, rejected = [], 0
        while len(rows) < SIZE:
            x = 2 * torch.rand((2,), generator=generator, dtype=torch.float32) - 1
            identity = tuple(x.tolist())
            if identity in seen:
                rejected += 1
                continue
            seen.add(identity)
            rows.append(x)
        x = torch.stack(rows)
        affine = x @ a - b
        periodic = torch.sin(2*torch.pi*(x @ q[0]) + phi[0]) + .6*torch.cos(2*torch.pi*(x @ q[1]) + phi[1])
        score = {'affine': affine, 'periodic': periodic, 'mixed': affine + .65*periodic}[regime]
        arrays.update({split+'_inputs': x, split+'_affine_score': affine,
                       split+'_periodic_score': periodic, split+'_teacher_score': score,
                       split+'_labels': (score > 0).long()})
        rejections[split] = rejected
    assert len(seen) == 2*SIZE
    return arrays, {'teacher_seed': teacher_seed, 'split_seeds': dict(zip(('train', 'dev'), seeds)),
                    'a_norm_rejections': a_rejections, 'frequency_rejections': q_rejections,
                    'duplicate_input_rejections': rejections}


def prepare_features(arrays, init_index):
    result = dict(arrays)
    generator = torch.Generator().manual_seed(42101 + init_index)
    projection = torch.randn((16, 2), generator=generator, dtype=torch.float32) * 1.5
    result['projection_B'] = projection
    for split in ('train', 'dev'):
        z = 2*torch.pi*(arrays[split+'_inputs'] @ projection.T)
        result[split+'_z'] = z
        result[split+'_fourier_raw'] = torch.cat((z.sin(), z.cos()), -1)
    raw = result['train_fourier_raw']
    result['fourier_train_mean'] = raw.mean(0)
    result['fourier_train_std'] = raw.std(0, unbiased=False).clamp_min(1e-6)
    for split in ('train', 'dev'):
        result[split+'_fourier_features'] = (result[split+'_fourier_raw'] - result['fourier_train_mean']) / result['fourier_train_std']
    return result


class RawDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 48)
        self.blocks = nn.ModuleList([nn.Linear(48, 48) for _ in range(2)])
        self.output = nn.Linear(48, 2)

    def forward(self, x):
        h = F.gelu(self.input(x))
        for block in self.blocks:
            h = h + F.gelu(block(h))
        return self.output(h)


class Classifier(nn.Module):
    def __init__(self, arm, reference, raw_reference, dense_reference):
        super().__init__()
        self.arm = arm
        if arm == 'raw_dense':
            self.dense = copy.deepcopy(dense_reference)
        elif arm == 'concat':
            self.head = nn.Sequential(nn.Linear(34, 32), nn.ReLU(), nn.Linear(32, 2))
            with torch.no_grad():
                self.head[0].weight[:, :32].copy_(reference[0].weight)
                self.head[0].weight[:, 32:].zero_()
                self.head[0].bias.copy_(reference[0].bias)
                self.head[2].load_state_dict(reference[2].state_dict())
        else:
            self.head = copy.deepcopy(reference)
            if arm != 'fourier':
                self.raw = copy.deepcopy(raw_reference)
                if arm in ('input_gate', 'constant_gate'):
                    self.gate = nn.Parameter(torch.zeros(3))

    def components(self, x):
        features, raw = x[:, :32], x[:, 32:]
        if self.arm == 'raw_dense':
            return self.dense(raw), {}
        if self.arm == 'concat':
            return self.head(x), {}
        fourier = self.head(features)
        if self.arm == 'fourier':
            return fourier, {'fourier_logits': fourier}
        direct = self.raw(raw)
        if self.arm == 'input_gate':
            gate = torch.sigmoid(raw @ self.gate[:2] + self.gate[2])
        elif self.arm == 'constant_gate':
            gate = torch.sigmoid(self.gate.sum() / (3 ** .5)).expand(raw.shape[0])
        else:
            gate = raw.new_full((raw.shape[0],), .5)
        logits = (1-gate[:, None])*direct + gate[:, None]*fourier
        return logits, {'gate': gate, 'raw_logits': direct, 'fourier_logits': fourier}

    def forward(self, x):
        return self.components(x)[0]


def activation_check(module, inputs, output):
    if isinstance(output, torch.Tensor):
        finite(output, type(module).__name__ + ' activation')


@torch.no_grad()
def evaluate(model, batches, known_train_loss=None):
    model.eval()
    result = {}
    for split, (features, labels) in batches.items():
        logits, components = model.components(features)
        loss = logits.new_tensor(known_train_loss) if split == 'train' and known_train_loss is not None else F.cross_entropy(logits, labels)
        finite(logits, split+' logits')
        finite(loss, split+' loss')
        diagnostics = {}
        for path in ('raw', 'fourier'):
            if path+'_logits' in components:
                finite(components[path+'_logits'], path+' logits')
                diagnostics[path+'_logit_rms'] = components[path+'_logits'].square().mean().sqrt().item()
        if 'gate' in components:
            gate = components['gate']
            finite(gate, 'gate')
            diagnostics.update({'gate_mean': gate.mean().item(), 'gate_std': gate.std(unbiased=False).item(),
                                'gate_min': gate.min().item(), 'gate_max': gate.max().item(),
                                'gate_coefficients': model.gate.detach().tolist() if hasattr(model, 'gate') else []})
        result[split] = {'loss': loss.item(), 'accuracy': (logits.argmax(-1) == labels).float().mean().item(),
                         'logit_rms': logits.square().mean().sqrt().item(), **diagnostics}
    return result


def config(arm, source_sha):
    return {'script': 'input_gated_fourier_classifier.py', 'source_sha256': source_sha, 'arm': arm,
            'architecture': 'R=Linear2,2; F=Linear32,32 ReLU Linear32,2; blend=(1-g)R+gF; input g=sigmoid(raw@gate[:2]+gate[2]); constant g=sigmoid(sum(gate)/sqrt3); static g=.5; concat=Linear34,32 ReLU Linear32,2; raw_dense residual GELU48x2',
            'input_layout': 'exact cat(saved split_fourier_features,saved split_inputs),dim=-1; no new transforms; Fourier32 then raw2',
            'external_fixed_state': 'B/fourier_train_mean/std in checkpoint-linked exact dataset path/hash; inference z=2*pi*x@B.T,cat(sin(z),cos(z)),standardize with saved train stats; raw_dense uses raw2 only',
            'trainable_parameters': PARAMETERS[arm], 'optimizer': 'AdamW', 'learning_rate': .003,
            'weight_decay': .001, 'betas': [.9, .999], 'epsilon': 1e-8, 'steps': STEPS,
            'batch': 'full', 'device': 'cpu', 'threads': 4, 'selection_observed_steps': list(range(STEPS+1))}


def fit(model, batches, writer, arm, fit_id, dataset, source_sha, init_seed, init_index):
    start = time.monotonic()
    hooks = [module.register_forward_hook(activation_check) for module in model.modules()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001, betas=(.9, .999), eps=1e-8)
    x, y = batches['train']
    loss = F.cross_entropy(model(x), y)
    finite(loss, 'state0 loss')
    diagnostics = {'0': evaluate(model, batches, known_train_loss=loss.item())}
    losses = [loss.item()]
    best = BestCheckpoint(scope='minimum full-training CE at actual states0..120; earliest tied minimum; no held-out selection')
    best.observe(model, losses[0], 0)
    for step in range(1, STEPS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        finite(loss, 'pre-update loss')
        loss.backward()
        for name, parameter in model.named_parameters():
            finite(parameter.grad, 'gradient '+name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            finite(parameter, 'post-update parameter '+name)
        model.eval()
        observed = F.cross_entropy(model(x), y)
        finite(observed, 'post-update loss')
        losses.append(observed.item())
        best.observe(model, observed.item(), step)
        loss = observed
        if step in DIAGNOSTIC_STEPS:
            diagnostics[str(step)] = evaluate(model, batches, known_train_loss=observed.item())
    assert best.observations == STEPS+1 and best.observed_steps == list(range(STEPS+1))
    configuration = {**config(arm, source_sha), 'fit_id': fit_id, 'init_seed': init_seed,
                     'projection_seed': 42101+init_index, 'raw_seed':43101+init_index,
                     'raw_dense_seed':44101+init_index, 'dataset_sha256':dataset['sha256']}
    checkpoint = best.save(writer, fit_id+'-'+arm+'-best', arm=arm, seed=init_seed,
                           config=configuration, datasets=[dataset['path']])
    assert checkpoint['selection']['value'] == min(losses)
    assert checkpoint['selection']['step'] == losses.index(min(losses))
    for hook in hooks:
        hook.remove()
    return {'training_loss_every_state':losses, 'diagnostic_states':diagnostics,
            'best_training_checkpoint':checkpoint['path'], 'best_training_selection':checkpoint['selection'],
            'final_update':STEPS, 'final_update_metrics':diagnostics[str(STEPS)],
            'elapsed_seconds':time.monotonic()-start, 'activations_gradients_parameters_finite':True}


def aggregate(records):
    metrics, comparisons = {}, {}
    contrasts = {'mixed':('constant_gate', 'static_blend', 'concat'), 'affine':('raw_dense',), 'periodic':('fourier',)}
    for regime in REGIMES:
        rows = [r for r in records if r['regime'] == regime]
        for arm in ARMS:
            metrics[f'{regime}_{arm}_dev_loss'] = sum(r['arms'][arm]['final_update_metrics']['dev']['loss'] for r in rows)/len(rows)
        comparisons[regime] = {}
        for control in contrasts[regime]:
            replicates = []
            for teacher_index in range(3):
                paired = [r for r in rows if r['teacher_index'] == teacher_index]
                differences = [r['arms']['input_gate']['final_update_metrics']['dev']['loss'] - r['arms'][control]['final_update_metrics']['dev']['loss'] for r in paired]
                replicates.append({'teacher_index':teacher_index, 'init_loss_differences':differences, 'mean_dev_loss_difference':sum(differences)/len(differences)})
            means = [r['mean_dev_loss_difference'] for r in replicates]
            comparisons[regime][control] = {'replicates':replicates, 'mean':sum(means)/len(means), 'min':min(means), 'max':max(means)}
            metrics[f'{regime}_input_minus_{control}_dev_loss'] = sum(means)/len(means)
    passed = (all(comparisons['mixed'][a]['mean'] <= -.01 and comparisons['mixed'][a]['max'] < 0 for a in ('constant_gate', 'static_blend'))
              and comparisons['mixed']['concat']['mean'] <= .01
              and comparisons['affine']['raw_dense']['mean'] <= .02
              and comparisons['periodic']['fourier']['mean'] <= .02)
    return metrics, comparisons, passed


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
            for init_index, init_seed in enumerate(INIT_SEEDS):
                payload = prepare_features(arrays, init_index)
                for name, value in payload.items():
                    finite(value, name)
                hashes = {name:digest(value) for name,value in payload.items()}
                provenance = {'kind':'generated analytic synthetic', 'generator':'input_gated_fourier_classifier.py:task/prepare_features',
                              'script_sha256':source_sha, 'regime':regime, 'teacher_index':j, **generation,
                              'projection_seed':42101+init_index, 'init_seed':init_seed,
                              'raw_seed':43101+init_index, 'raw_dense_seed':44101+init_index,
                              'draw_order':'CPU float32; teacher a randn2 normalize reject norm<1e-6; b=.3*rand()-.15; two q randn2*1.5 reject norms outside[.75,3]; phi=2*pi*rand2. Coordinates train/dev independent generators, one rand2*2-1 row at a time, reject exact duplicates across splits',
                              'teacher':'A=x@a-b; P=sin(2*pi*x@q1+phi1)+.6*cos(2*pi*x@q2+phi2); affine A,periodic P,mixed A+.65P; labels score>0',
                              'split':{'train':SIZE,'dev':SIZE,'test':'not generated'},
                              'label_counts':{sp:payload[sp+'_labels'].bincount(minlength=2).tolist() for sp in ('train','dev')},
                              'array_sha256':hashes, 'feature_preparation':'B16x2 iid Normal(0,1.5^2); z=2*pi*x@B.T; cat(sin(z),cos(z)); train mean/population std clamp_min1e-6; raw coordinates unstandardized',
                              'downloads':False, 'generator_software':{'torch':str(torch.__version__),'device':'cpu','dtype':'float32'}}
                fit_id = f'{regime}-teacher-{j}-init-{init_seed}'
                dataset = writer.save_dataset(fit_id, payload, provenance=provenance)
                prepared.append((regime,j,init_index,init_seed,payload,dataset,hashes,generation,fit_id))
    assert len(writer.manifest['datasets']) == 18 and not writer.manifest['checkpoints']
    records = []
    for regime,j,init_index,init_seed,payload,dataset,hashes,generation,fit_id in prepared:
        torch.manual_seed(init_seed)
        reference = nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
        torch.manual_seed(43101+init_index)
        raw_reference = nn.Linear(2,2)
        torch.manual_seed(44101+init_index)
        dense_reference = RawDense()
        models = {arm:Classifier(arm,reference,raw_reference,dense_reference) for arm in ARMS}
        assert {a:sum(p.numel() for p in m.parameters()) for a,m in models.items()} == PARAMETERS
        assert all(all(torch.equal(v,models[a].head.state_dict()[n]) for n,v in reference.state_dict().items()) for a in ('input_gate','constant_gate','static_blend','fourier'))
        assert all(all(torch.equal(v,models[a].raw.state_dict()[n]) for n,v in raw_reference.state_dict().items()) for a in ('input_gate','constant_gate','static_blend'))
        order = ARMS if len(records)%2 == 0 else tuple(reversed(ARMS))
        row = {'regime':regime,'teacher_index':j,'init_seed':init_seed,'projection_seed':42101+init_index,
               'raw_seed':43101+init_index,'raw_dense_seed':44101+init_index,'generation':generation,
               'dataset':dataset['path'],'dataset_sha256':dataset['sha256'],'array_sha256':hashes,
               'initial_state_sha256':{a:{n:digest(v) for n,v in m.state_dict().items()} for a,m in models.items()},
               'trainable_parameters':PARAMETERS,'arm_order':order,'arms':{},
               'fixed_projection_elements':32,'normalization_elements':64,
               'projection_frequency_norms':payload['projection_B'].norm(dim=1).tolist(),
               'feature_diagnostics':{'train_mean':payload['train_fourier_features'].mean(0).tolist(),
                                      'train_std':payload['train_fourier_features'].std(0,unbiased=False).tolist(),
                                      'train_min':payload['train_fourier_features'].min(0).values.tolist(),
                                      'train_max':payload['train_fourier_features'].max(0).values.tolist()}}
        batches = {sp:(torch.cat((payload[sp+'_fourier_features'],payload[sp+'_inputs']),-1),payload[sp+'_labels']) for sp in ('train','dev')}
        for arm in order:
            row['arms'][arm] = fit(models[arm],batches,writer,arm,fit_id,dataset,source_sha,init_seed,init_index)
        records.append(row)
    assert len(records) == 18 and len(writer.manifest['checkpoints']) == 108
    metrics,comparisons,passed = aggregate(records)
    arm_summaries = {}
    for regime in REGIMES:
        arm_summaries[regime] = {}
        for arm in ARMS:
            endpoints = {}
            for endpoint in ('loss', 'accuracy'):
                teachers = []
                for teacher_index in range(3):
                    paired = [row for row in records if row['regime'] == regime and row['teacher_index'] == teacher_index]
                    values = [row['arms'][arm]['final_update_metrics']['dev'][endpoint] for row in paired]
                    teachers.append({'teacher_index':teacher_index, 'init_seeds':[row['init_seed'] for row in paired],
                                     'init_values':values, 'mean':sum(values)/len(values)})
                means = [teacher['mean'] for teacher in teachers]
                endpoints[endpoint] = {'teachers':teachers, 'teacher_min':min(means), 'teacher_max':max(means),
                                       'regime_mean':sum(means)/len(means)}
            arm_summaries[regime][arm] = endpoints
    details = {'status':'completed','script_sha256':source_sha,'elapsed_seconds':time.monotonic()-started,
               'config':{'regimes':REGIMES,'teacher_indices':[0,1,2],'init_seeds':INIT_SEEDS,'arms':{a:config(a,source_sha) for a in ARMS},
                         'primary_endpoint':'independent final120 dev CE; pair2inits then equal3teachers',
                         'screening_rule':'mixed input-minus constant/static mean<=-.01 and all3teacher differences<0; mixed concat mean<=.01; affine raw_dense mean<=.02; periodic Fourier mean<=.02',
                         'stopping_rule':'exactly120 updates; no dev selection,tuning,test'},
               'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':torch.get_num_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
               'records':records,'per_arm_final_dev_summary':arm_summaries,'paired_final_dev_contrasts':comparisons,'screening_criterion_met':passed,
               'limitations':'Development-only synthetic finite-budget screen,3teachers/regime and2coupledprojection/headseeds. Established Fourier/hybrid/gating ingredients; no novelty,significance,equal compute/tuning,broad superiority or mechanism attribution. Logit blending alters confidence/calibration. Constant redundant gate matches initial aggregate gradient norm only, NOT coordinatewise AdamW geometry or effective step; lower function capacity. Concat1186 versus1131 gated versus4946 dense parameters. No test,transfer,extrapolation; failure stops without rescue.'}
    summary = 'Frozen input-gated raw-linear/Fourier dev screen:3regimes x3teachers x2pairedinitializations x6arms,108fits120CPUupdates. Input/constant/static gate,concat,Fourier,raw_dense.18exact datasets,108minimum-training-CE checkpoints from121states. Final120dev independent; no test,tuning,promotion.'
    assert 1 <= len(metrics) <= 32 and len(summary) <= 2000
    output = Path('/output')
    (output/'details.json').write_text(json.dumps(details,allow_nan=False,indent=2))
    result = json.dumps({'metrics':metrics,'summary':summary},allow_nan=False,indent=2)
    (output/'result.json').write_text(result)
    print(result)


if __name__ == '__main__':
    main()
