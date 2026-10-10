import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

ml, remote, profiles = (module(n) for n in ('model_layout', 'remote', 'prepare_profiles'))


class LocalNodes:
    """Exercise real worker/config files and controller, faking only services/network."""
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.services = {}
        self.states = {}
        self.fail_configure = None
        for name, node in config['nodes'].items():
            for s in node.get('services', [node.get('service')]):
                self.services[s] = True
            if node['kind'] == 'proxy':
                self.reload(node)

    def reload(self, node):
        self.states[node['config']] = {
            name: {**spec, 'name': name, 'status': 'running' if spec.get('autoload') else 'stopped'}
            for name, spec in remote.read_config(node)['runtimes'].items()}

    def service(self, action, service):
        self.services[service] = action == 'start'
        for n in self.config['nodes'].values():
            if n.get('service') == service:
                if action == 'start':
                    self.reload(n)
                else:
                    for r in self.states[n['config']].values(): r['status'] = 'stopped'

    def call(self, name, op, **kwargs):
        self.calls.append((name, op, kwargs))
        n = self.config['nodes'][name]
        with patch.object(remote, 'active', side_effect=lambda s: self.services[s]), \
             patch.object(remote, 'systemctl', side_effect=self.service), \
             patch.object(remote, 'runtime_states', side_effect=lambda n: self.states[n['config']]), \
             patch.object(remote, 'ready', side_effect=lambda r: r['status'] == 'running'), \
             patch.object(remote, 'busy', return_value=False), \
             patch.object(remote, 'check_backend'):
            if op == 'diagnostics':
                return {k: {'ready': self.states[n['config']][k]['status'] == 'running', 'failed': False, 'status': 'ready'} for k in kwargs['names']}
            result = remote.dispatch({'node': n, 'op': op, **kwargs})
            if op == 'configure' and self.fail_configure == name:
                self.fail_configure = None
                raise RuntimeError('Injected failure after character write')
            return result


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = yaml.safe_load((HERE / 'layouts.yaml').read_text())
        self.config.pop('coordinator_hostname')
        self.files = []
        for name, node in self.config['nodes'].items():
            node['local'] = True
            node['state_dir'] = str(self.root / name / 'state')
            if node['kind'] == 'proxy':
                node['service'] = name + '.service'
                node['config'] = str(self.root / (name + '.yaml'))
                node['required_paths'] = []
                node['python_checks'] = []
                data = {'runtimes': {k: {'driver': 'llama_cpp', 'autoload': k in self.config['modes']['27b'][name], 'on_demand': False} for k in node['managed'] if k not in node['additions'] or k in self.config['modes']['27b'][name]}, 'custom': 'preserved', 'hosts': {'remote-173': {'token': 'test'}, 'main-61': {'token': 'test'}}}
                Path(node['config']).write_text(yaml.safe_dump(data))
                self.files.append(Path(node['config']))
            else:
                node['directory'] = str(self.root / 'nemor')
                for sub in ['characters','model_profiles']:
                    (Path(node['directory']) / sub).mkdir(parents=True)
                for char, modes in node['characters'].items():
                    path = Path(node['directory']) / 'characters' / (char + '.json')
                    path.write_text(json.dumps({'llm_profile': modes['27b'], 'custom': ['keep'], 'context_size': 131072}))
                    self.files.append(path)
                for name, expected in node['expected_profiles'].items():
                    (Path(node['directory']) / 'model_profiles' / (name + '.json')).write_text(json.dumps(expected))
        self.transport = LocalNodes(self.config)
        self.controller = ml.Controller(self.config, self.transport, self.root / 'controller', output=lambda *a, **k: None)

    def test_four_modes_round_trip_and_noop(self):
        for mode in ['flash-ring','27b-ring','flash','27b']:
            with self.subTest(mode=mode):
                self.controller.switch(mode)
                state = self.controller.status()['nodes']
                self.assertTrue(self.controller.matches(mode, state))
                self.transport.calls.clear()
                self.controller.switch(mode)
                self.assertTrue(all(op == 'preflight' for _, op, _ in self.transport.calls))
                for path in self.files:
                    data = yaml.safe_load(path.read_text())
                    self.assertIn('custom', data)
                for node in self.config['nodes'].values():
                    if node['kind'] == 'nemor':
                        for char in node['characters']:
                            d=json.loads((Path(node['directory'])/'characters'/(char+'.json')).read_text())
                            self.assertEqual(d.get('context_management'), 'runner' if mode.endswith('-ring') else None)
                            if mode == '27b' and char == 'tet': self.assertNotIn('request_timeout', d)
        gpu = yaml.safe_load(Path(self.config['nodes']['gpu173']['config']).read_text())
        self.assertFalse(gpu['runtimes']['vcortex-shadow']['on_demand'])

    def test_plan_is_read_only_and_shows_settings(self):
        before = {p:p.read_bytes() for p in self.files}
        plan = self.controller.plan('flash-ring')
        self.assertEqual(plan['character_settings']['tet']['context_management'], 'runner')
        self.assertEqual(before, {p:p.read_bytes() for p in self.files})
        self.assertFalse((self.root / 'controller').exists())
        self.assertTrue(all(op == 'preflight' for _, op, _ in self.transport.calls))

    def test_settings_drift_prevents_noop(self):
        self.controller.switch('flash-ring')
        state = self.controller.status()['nodes']
        state['nemor']['character_settings']['tet']['context_management'] = None
        self.assertFalse(self.controller.matches('flash-ring', state))
        state = self.controller.status()['nodes']
        state['aurora']['inference_settings']['request_timeout_seconds'] = 300
        self.assertFalse(self.controller.matches('flash-ring', state))

    def test_failed_switch_restores_exact_original_files(self):
        before = {p:p.read_bytes() for p in self.files}
        self.transport.fail_configure = 'nemor'
        with self.assertRaisesRegex(ml.LayoutError, 'previous state restored'):
            self.controller.switch('flash-ring')
        self.assertEqual(before, {p:p.read_bytes() for p in self.files})
        self.assertEqual(self.controller.read_journal()['status'], 'rolled_back')
        self.assertTrue(all(self.transport.services.values()))

    def test_concurrent_edit_is_not_overwritten(self):
        node = self.config['nodes']['nemor']
        self.transport.call('nemor','lock',tx='test')
        self.transport.call('nemor','snapshot',tx='test',modes=list(self.config['modes']))
        for s in node['services']: self.transport.services[s] = False
        self.transport.call('nemor','configure',tx='test',mode='flash-ring')
        p = Path(node['directory'])/'characters/atom.json'
        p.write_text('{"external":true}')
        with self.assertRaisesRegex(RuntimeError, 'Concurrent edit'):
            self.transport.call('nemor','restore',tx='test')
        self.assertEqual(json.loads(p.read_text()), {'external':True})

    def test_incomplete_reverse_settings_rejected(self):
        del self.config['nodes']['nemor']['character_settings']['27b']['tet']['context_management']
        with self.assertRaisesRegex(ml.LayoutError, 'same character fields'):
            ml.validate(self.config)

    def test_profile_preparation_preserves_credentials_and_existing_files(self):
        base = self.root/'profiles/model_profiles'
        base.mkdir(parents=True)
        for target,(source,server,native,model) in profiles.PROFILES.items():
            (base/(source+'.json')).write_text(json.dumps({'id':source,'server':server,'model':native,'auth_token':'private-test-token','tls_fingerprint':'test-pin'}))
        profiles.prepare(base.parent)
        original = {p:p.read_bytes() for p in base.iterdir()}
        profiles.prepare(base.parent)
        self.assertEqual(original, {p:p.read_bytes() for p in base.iterdir()})
        for target,(_,_,_,model) in profiles.PROFILES.items():
            p=base/(target+'.json'); d=json.loads(p.read_text())
            self.assertEqual(d['model'],model)
            self.assertEqual(d['auth_token'],'private-test-token')
            self.assertEqual(d['tls_fingerprint'],'test-pin')
            self.assertEqual(p.stat().st_mode & 0o777,0o600)
        p=base/'model-flash-ring.json'; p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'refusing overwrite'): profiles.prepare(base.parent)


if __name__ == '__main__': unittest.main()
