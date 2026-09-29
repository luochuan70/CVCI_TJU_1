from __future__ import absolute_import

import json
import os
import sys
import unittest
import xml.etree.ElementTree as ET

CODES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.dirname(CODES_ROOT)
if CODES_ROOT not in sys.path:
    sys.path.insert(0, CODES_ROOT)

from harness.ordinary_classifier import OrdinarySceneClassifier44
from ordinary_rules.configs import get_config
from ordinary_rules.router import OrdinaryRuleRouter


class PackageRuntimeTests(unittest.TestCase):
    def test_every_ordinary_class_has_a_rule(self):
        router = OrdinaryRuleRouter(
            model_path=os.path.join(CODES_ROOT, "data", "classifier_model.json")
        )
        self.assertEqual(sorted(router.rules), list(range(1, 45)))
        for class_id in range(1, 45):
            self.assertEqual(get_config(class_id)["class_id"], class_id)
        router.close()

    def test_classifier_model_loads_from_package(self):
        path = os.path.join(CODES_ROOT, "data", "classifier_model.json")
        classifier = OrdinarySceneClassifier44(model_path=path)
        self.assertTrue(classifier.model)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(
            classifier.model.get("feature_names"), data.get("feature_names")
        )

    def test_official_route_counts(self):
        expected = {
            "bench2drive220.xml": 220,
            "CVCI_BenchMark.xml": 144,
        }
        for name, count in expected.items():
            root = ET.parse(os.path.join(PACKAGE_ROOT, "route", name)).getroot()
            routes = [
                node for node in root.iter()
                if node.tag.rsplit("}", 1)[-1] == "route"
            ]
            self.assertEqual(len(routes), count)


if __name__ == "__main__":
    unittest.main()
