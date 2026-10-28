import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch
from torch import nn

from neural_lab.visualization import VisualizationError, describe_model, export_visualizations


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x) + x


class VisualizationTests(unittest.TestCase):
    def test_residual_connections_shapes_and_unique_parameter_total(self):
        graph = describe_model(Residual(), (torch.randn(2, 4),))
        add = next(node for node in graph["nodes"] if node["label"] == "add")
        self.assertEqual(len([edge for edge in graph["edges"] if edge["target"] == add["id"]]), 2)
        self.assertEqual(add["output"]["shape"], [2, 4])
        self.assertEqual(graph["parameters"], 20)
        self.assertEqual(graph["trainable_parameters"], 20)

    def test_shared_module_total_is_not_double_counted(self):
        class Shared(Residual):
            def forward(self, x):
                return self.linear(x) + self.linear(x)
        graph = describe_model(Shared(), (torch.randn(2, 4),))
        calls = [node for node in graph["nodes"] if node["op"] == "call_module"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(graph["parameters"], 20)

    def test_preserves_original_modes_buffers_gradients_inputs_and_rng(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(), nn.Linear(4, 2))
        model.train()
        model[1].eval()
        modes = [module.training for module in model.modules()]
        model[2].weight.grad = torch.ones_like(model[2].weight)
        gradient = model[2].weight.grad.clone()
        running_mean = model[0].running_mean.clone()
        inputs = torch.randn(3, 4)
        before_input = inputs.clone()
        rng = torch.get_rng_state().clone()
        describe_model(model, (inputs,))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(model[0].running_mean, running_mean))
        self.assertTrue(torch.equal(model[2].weight.grad, gradient))
        self.assertTrue(torch.equal(inputs, before_input))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_activation_sample_bounded_and_nonfinite_values_are_json_safe(self):
        graph = describe_model(nn.Identity(), (torch.randn(4, 256),))
        sample = graph["nodes"][0]["activation_sample"]
        self.assertEqual(sample["sample_size"], 64)
        self.assertEqual(len(sample["values"]), 64)
        graph = describe_model(nn.Identity(), (torch.tensor([float("inf"), float("nan"), 1.0]),))
        sample = graph["nodes"][0]["activation_sample"]
        self.assertEqual(sample["values"], [None, None, 1.0])
        json.dumps(graph, allow_nan=False)

    def test_nested_outputs_preserve_shapes(self):
        class Pair(nn.Module):
            def forward(self, x):
                return {"first": x, "second": (x + 1, x.mean())}
        graph = describe_model(Pair(), (torch.randn(2, 4),))
        output = graph["nodes"][-1]["output"]
        self.assertEqual(output["first"]["shape"], [2, 4])
        self.assertEqual(output["second"][1]["shape"], [])

    def test_exports_consistent_artifacts_and_escapes_untrusted_title(self):
        title = '</script><img src=x onerror="alert(1)"> & <network>'
        with tempfile.TemporaryDirectory() as directory:
            paths = export_visualizations(Residual(), (torch.ones(2, 4),), directory, title=title)
            self.assertEqual(set(paths), {"architecture.json", "architecture.svg", "architecture.dot", "index.html"})
            graph = json.loads(paths["architecture.json"].read_text())
            ET.fromstring(paths["architecture.svg"].read_text())
            page = paths["index.html"].read_text()
            self.assertNotIn(title, page)
            self.assertIn("\\u003c/script\\u003e", page)
            self.assertIn("connect-src 'none'", page)
            self.assertIn('rankdir="LR"', paths["architecture.dot"].read_text())
            self.assertEqual(graph["title"], title)
            self.assertEqual(graph["nodes"][-1]["output"]["shape"], [2, 4])

    def test_data_dependent_control_flow_reports_failure_without_fake_diagram(self):
        class Dynamic(nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncreated"
            with self.assertRaises(VisualizationError):
                export_visualizations(Dynamic(), (torch.ones(2),), output)
            self.assertFalse(output.exists())

    def test_node_budget_rejects_large_graph_before_execution(self):
        with self.assertRaises(VisualizationError):
            describe_model(Residual(), (torch.ones(2, 4),), max_nodes=2)
        with self.assertRaises(TypeError):
            describe_model(nn.Identity(), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
