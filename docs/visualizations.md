# Network visualizations

Each visualization export produces:

| Artifact | Purpose |
| --- | --- |
| index.html | Offline interactive graph with search, zoom, layer inspection, and activation sample heatmaps |
| architecture.svg | Shareable layer diagram with tensor shapes and parameter counts |
| architecture.json | Structured graph, shapes, unique parameter totals, and sampled activations |
| architecture.dot | Editable Graphviz source for alternative layouts |

No JavaScript libraries, CDN requests, web server, Graphviz installation, or extra
Python dependencies are needed. PyTorch FX supplies the graph. HTML has embedded
data and works when opened directly from disk.

## Generate examples inside the sandbox

After the usual setup, run:

```bash
bash scripts/visualize-demo.sh
```

This uses a disposable Compose container with the normal GPU/resource restrictions
and two additional read-only mounts containing the reviewed visualization package
and examples. It leaves the base sandbox unchanged and creates:

```text
experiments/visualizations/
  mlp/
  cnn/
  residual-mlp/
```

Open any directory's index.html. SVG files can be embedded in reports and model
cards. Do not serve the whole experiment directory on your network.

## Export an experiment

Invoke this from training code inside the sandbox, where neural_lab is available
on PYTHONPATH through the visualization Compose overlay:

```python
import torch
from neural_lab.visualization import export_visualizations

paths = export_visualizations(
    model,
    (torch.randn(2, 16, device=next(model.parameters()).device),),
    "/workspace/my-run/visualizations",
    title="Experiment 001: residual MLP",
)
```

Use representative inputs matching your model's signature and device. Multiple
positional inputs belong in the tuple. A traceable wrapper can adapt keyword-only
interfaces. CPU models and inputs are recommended for exports: deepcopying a model
temporarily increases memory usage, including VRAM for CUDA models.

The exporter copies the model and inputs, switches the copy to eval mode, traces
it, and runs one forward pass without gradients. Original module modes, parameters,
buffers, and gradients remain intact. Torch RNG state is restored for the devices
used by model parameters/buffers and top-level tensor inputs.

This is real model execution, including custom Python, so never run this exporter
on untrusted experiment code outside the sandbox.

## Reading the views

Nodes include modules, function calls, method calls, inputs, and outputs. Residual
and other branch dependencies retain their edges. Per-node parameter counts can
repeat when a module is reused; the report's model total counts unique parameters.

Activation heatmaps contain at most 64 evenly sampled elements of the first
dense real-valued tensor in a node's output. The displayed min, max, and mean
describe that sample, not the full tensor. These are example-dependent values,
not neuron importance scores or full feature maps. Non-finite values are gray.

Shapes and activations represent one eval-mode example. This is not a training
timeline, a neuron-level 3D simulation, or proof that all execution paths were
traced. Dynamic Python control flow based on tensor values is not supported by
FX symbolic tracing. Unsupported models fail explicitly instead of receiving an
invented sequential diagram. The default node budget is 500.

Samples may reveal information about the example input. Use synthetic examples or
review data before publishing. Parameter values and raw checkpoints are not included.

The HTML renderer escapes text and embedded JSON, and declares an offline content
security policy. A future publisher must still validate artifacts coming from
workers: untrusted code can replace generated files.

## Research-loop integration

The demo worker exports visualizations after training, and the API/Codex planners
are instructed to attempt the same export. The supervisor journals worker-reported
findings. A future publishing step should include SVG/JSON and reviewed HTML
alongside model artifacts. Visualization failure must be recorded without
mislabeling a trained model as a failed training run. The Hugging Face publisher
is not implemented yet.

## Planned: julrog/nn_vis integration

Add [julrog/nn_vis](https://github.com/julrog/nn_vis) as an optional 3D
visualization backend alongside the existing FX diagrams. This integration is
planned; no upstream code or dependencies are currently included.

- [ ] Review the upstream license, citation requirements, dependencies, and
  supported model types. Pin a compatible upstream revision.
- [ ] Prototype a conversion from a small trained PyTorch MLP into the visualizer's
  network and importance-processing format. Document unsupported architectures;
  do not substitute our sampled activation values for its importance estimates.
- [ ] Run preprocessing with explicit time and memory limits inside a separate
  sandbox. Keep account credentials and host personal files outside it.
- [ ] Check headless rendering support for unattended local GPU runs. Save 3D
  snapshots or short clips as experiment artifacts; keep desktop interaction
  optional and avoid granting workers access to the host display.
- [ ] Store outputs under each run's `visualizations/julrog/` directory, with the
  upstream revision, processing parameters, seeds, and model provenance.
- [ ] Record unsupported models and rendering failures without failing training.
  Continue producing the existing HTML/SVG/JSON/DOT artifacts.

The first milestone is a reproducible 3D visualization of one trained MLP, with a
documented conversion and sandboxed rendering workflow. Broader model support
comes after that proof of integration.

## Other tools considered

- [aharley/nn_vis](https://github.com/aharley/nn_vis) demonstrates interactive
  digit-recognition networks, including MLP/CNN activations.
- [julrog/nn_vis](https://github.com/julrog/nn_vis) provides a dedicated 3D
  rendering and importance-processing pipeline.
- [Netron](https://github.com/lutzroeder/netron) is useful for inspecting exported
  ONNX models, and can be added as an optional local viewer later.

The initial exporter uses FX for portable diagrams across varied PyTorch models.
It does not vendor or claim compatibility with nn_vis's model formats.

## Validation

Run CPU checks in a trusted development checkout with PyTorch installed:

```bash
python -m unittest discover -s tests -v
python -m examples.visualize_models --output /tmp/neural-lab-visualizations
```

GitHub Actions runs these checks, validates Compose configuration, and uploads
the MLP/CNN/residual reports as example-network-visualizations. These checks do
not validate the target laptop's NVIDIA driver or GPU isolation.
