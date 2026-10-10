#!/usr/bin/env python3
"""Independent model placement controller. Run on the configured coordinator."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import yaml

HERE = Path(__file__).resolve().parent


class LayoutError(RuntimeError):
    pass


class ReadinessError(LayoutError):
    """A runtime has exited; waiting longer cannot make it ready."""


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate(c):
    if c.get('version') != 1:
        raise LayoutError('Unsupported configuration version')
    nodes, modes = c['nodes'], c['modes']
    if not modes or len([n for n in nodes.values() if n['kind'] == 'nemor']) != 1:
        raise LayoutError('Configure modes and exactly one Nemor node')
    for name, n in nodes.items():
        if n['kind'] not in ('proxy', 'nemor'):
            raise LayoutError('Unknown node kind: ' + name)
        if not n.get('local') and (not n.get('ssh') or n['ssh'].startswith('-')):
            raise LayoutError('Missing/invalid SSH destination: ' + name)
        for field in ('character_settings', 'inference_settings'):
            if field in n and set(n[field]) != set(modes):
                raise LayoutError('Settings must cover every mode: ' + name + '/' + field)
        if n['kind'] == 'proxy':
            for settings in n.get('inference_settings', {}).values():
                if set(settings) - {'request_timeout_seconds'}:
                    raise LayoutError('Unsupported inference setting')
                timeout = settings.get('request_timeout_seconds')
                if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
                    raise LayoutError('Invalid inference timeout')
            if len(set(n['managed'])) != len(n['managed']):
                raise LayoutError('Duplicate managed runtime: ' + name)
            for mode, spec in modes.items():
                if name not in spec or not set(spec[name]) <= set(n['managed']):
                    raise LayoutError('Invalid placement: ' + mode + '/' + name)
                if len(spec[name]) != len(set(spec[name])):
                    raise LayoutError('Duplicate placement: ' + mode + '/' + name)
        else:
            if not n.get('services') or not n.get('characters'):
                raise LayoutError('Nemor needs services and character mappings')
            for character, mapping in n['characters'].items():
                if set(mapping) != set(modes):
                    raise LayoutError('Incomplete character mapping: ' + character)
                for value in [character, *mapping.values()]:
                    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', value):
                        raise LayoutError('Invalid character/profile ID')
            allowed = {'context_management', 'send_session_id', 'request_timeout', 'context_size'}
            schemas = []
            for settings in n.get('character_settings', {}).values():
                if set(settings) != set(n['characters']):
                    raise LayoutError('Settings must cover every managed character')
                schemas.append({c: set(v) for c, v in settings.items()})
                for values in settings.values():
                    if set(values) - allowed:
                        raise LayoutError('Unsupported character setting')
                    if values.get('context_management') not in (None, 'runner'):
                        raise LayoutError('Invalid context management')
                    for key in ('request_timeout', 'context_size'):
                        v = values.get(key)
                        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
                            raise LayoutError('Invalid character setting: ' + key)
                    if 'send_session_id' in values and not isinstance(values['send_session_id'], bool):
                        raise LayoutError('send_session_id must be boolean')
            if schemas and any(s != schemas[0] for s in schemas):
                raise LayoutError('Every mode must set or clear the same character fields')
    return c


def check_route_cycles(config, states, mode):
    urls = {n['api_url'].rstrip('/'): name for name, n in config['nodes'].items() if n.get('api_url')}
    routes = {name: {**states[name].get('routes', {}), **node.get('routes', {}).get(mode, {})}
              for name, node in config['nodes'].items() if node['kind'] == 'proxy'}

    for name, node in config['nodes'].items():
        for route in node.get('removed_routes', []):
            routes.get(name, {}).pop(route, None)

    def visit(key, path):
        if key in path:
            raise LayoutError('Routing cycle in mode ' + mode + ': ' + ' -> '.join(n + '/' + r for n, r in [*path, key]))
        node, route = key
        for backend in routes[node].get(route, {}).get('backends', []):
            target = urls.get(backend['url'].rstrip('/'))
            if target:
                visit((target, backend.get('model', route)), [*path, key])

    for node, entries in routes.items():
        for route in entries:
            visit((node, route), [])


class Transport:
    def __init__(self, config):
        self.config = config
        self.worker = (HERE / 'remote.py').read_text()

    def call(self, node_name, op, **params):
        name = node_name
        node = self.config['nodes'][name]
        payload = {'node': node, 'op': op, **params}
        if node.get('local'):
            cmd = [node.get('python', 'python3'), '-c', self.worker]
        else:
            cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                   node['ssh'], shlex.join([node.get('python', 'python3'), '-c', self.worker])]
        try:
            p = subprocess.run(cmd, input=json.dumps(payload), text=True,
                               capture_output=True, timeout=1250 if op == 'runtime' else 110)
        except subprocess.TimeoutExpired:
            raise LayoutError(f'{name}: {op}: transport timeout; operation may have completed') from None
        try:
            data = json.loads(p.stdout)
        except ValueError:
            raise LayoutError(f'{name}: {op}: invalid SSH/worker response (exit {p.returncode})') from None
        if not data.get('ok') or p.returncode:
            raise LayoutError(f"{name}: {op}: {data.get('error', 'worker failed')}")
        return data['result']


class Controller:
    def __init__(self, config, transport=None, state_dir=None, output=print):
        self.config = validate(config)
        self.transport = transport or Transport(config)
        self.directory = Path(state_dir or config['state_dir']).expanduser()
        self.path = self.directory / 'transaction.json'
        self.output = output
        self.proxies = [n for n, spec in config['nodes'].items() if spec['kind'] == 'proxy']
        self.nemor = next(n for n, spec in config['nodes'].items() if spec['kind'] == 'nemor')
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.journal = None

    def call(self, node_name, op, **params):
        name = node_name
        return self.transport.call(name, op, **params)

    def read_journal(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self):
        atomic_json(self.path, self.journal)
        atomic_json(self.directory / 'history' / (self.journal['id'] + '.json'), self.journal)

    def event(self, message):
        self.output(message, flush=True)
        self.journal['events'].append({'time': time.time(), 'message': message})
        self.save()

    def action(self, node_name, op, **params):
        name = node_name
        self.event(f'{name}: {op}' + (f" {params['action']} {params['service']}" if op == 'service' else ''))
        self.journal['pending'] = {'node': name, 'op': op, **params}
        self.save()
        result = self.call(name, op, tx=self.journal['id'], **params)
        self.journal['pending'] = None
        self.save()
        return result

    @contextlib.contextmanager
    def exclusive(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / 'controller.lock').open('a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LayoutError('A switch/recovery is already running') from None
            yield

    def check_coordinator(self):
        expected = self.config.get('coordinator_hostname')
        if expected and socket.gethostname().lower() != expected.lower():
            raise LayoutError(f'Run this configuration on {expected}; this is {socket.gethostname()}')

    def status(self):
        self.check_coordinator()
        return {'nodes': {n: self.call(n, 'inspect') for n in self.config['nodes']},
                'transaction': self.read_journal()}

    def preflight(self):
        self.check_coordinator()
        return {n: self.call(n, 'preflight', modes=list(self.config['modes'])) for n in self.config['nodes']}

    def matches(self, mode, states):
        for name in self.proxies:
            n = self.config['nodes'][name]
            current = states[name]
            for key, value in n.get('inference_settings', {}).get(mode, {}).items():
                if current.get('inference_settings', {}).get(key) != value:
                    return False
            desired = set(self.config['modes'][mode][name])
            if not current['services'][n['service']]:
                return False
            for runtime in n['managed']:
                r = current['runtimes'].get(runtime)
                if not r or r['status'] != ('running' if runtime in desired else 'stopped'):
                    return False
                if r.get('on_demand') or r.get('autoload') != (runtime in desired):
                    return False
                if runtime in desired and not r.get('ready'):
                    return False
            if any(r in current.get('routes', {}) for r in n.get('removed_routes', [])):
                return False
            for route, spec in n.get('routes', {}).get(mode, {}).items():
                actual = current.get('routes', {}).get(route, {})
                expected = {'kind': spec['kind'], 'health_path': spec.get('health_path', '/health'), 'backends': [
                    {**{k: b[k] for k in ('url', 'model', 'auth') if k in b},
                     'tls_verify': b.get('tls_verify', b.get('verify_tls', True))} for b in spec['backends']]}
                if actual != expected:
                    return False
        characters = states[self.nemor]['characters']
        for name, settings in self.config['nodes'][self.nemor].get('character_settings', {}).get(mode, {}).items():
            actual = states[self.nemor].get('character_settings', {}).get(name, {})
            if any(actual.get(k) != v for k, v in settings.items()):
                return False
        return all(characters.get(k) == v[mode] for k, v in self.config['nodes'][self.nemor]['characters'].items())

    def plan(self, mode):
        if mode not in self.config['modes']:
            raise LayoutError('Unknown mode: ' + mode)
        states = self.preflight()
        for candidate in self.config['modes']:
            check_route_cycles(self.config, states, candidate)
        return {'mode': mode, 'already_active': self.matches(mode, states), 'placement': self.config['modes'][mode],
                'characters': {k: v[mode] for k, v in self.config['nodes'][self.nemor]['characters'].items()},
                'character_settings': self.config['nodes'][self.nemor].get('character_settings', {}).get(mode, {}),
                'inference_settings': {n: self.config['nodes'][n].get('inference_settings', {}).get(mode, {}) for n in self.proxies},
                'current': states,
                'steps': ['lock and snapshot every node', 'stop voice clients; wait for idle Nemor; stop Nemor',
                          'stop both model proxies', 'write placement, routes and character profiles',
                          'start proxies; check every target runtime', 'start previously active Nemor/voice services']}

    def wait_for(self, description, check, timeout):
        deadline = time.monotonic() + timeout
        last_report = 0
        last_error = None
        while True:
            try:
                if check():
                    return
            except ReadinessError:
                raise
            except (LayoutError, OSError) as e:
                last_error = str(e)
            if time.monotonic() >= deadline:
                raise LayoutError(description + ': timeout' + (f' ({last_error})' if last_error else ''))
            if time.monotonic() - last_report >= 20:
                self.output('Waiting: ' + description, flush=True)
                last_report = time.monotonic()
            time.sleep(2)

    def wait_models(self, name, names):
        stopped_checks = {}
        last_report = [0]

        def check():
            state = self.call(name, 'diagnostics', names=names)
            for runtime, info in state.items():
                stopped_checks[runtime] = (stopped_checks.get(runtime, 0) + 1
                                           if info['failed'] else 0)
                if stopped_checks[runtime] >= 2:
                    raise ReadinessError(f'{name}: {runtime} exited during startup; '
                                         f'check llm-proxy logs {runtime} on that host')
            if all(info['ready'] for info in state.values()):
                return True
            if time.monotonic() - last_report[0] >= 20:
                description = ', '.join(f"{r}={i['status']}" for r, i in state.items())
                self.output(name + ': ' + description, flush=True)
                last_report[0] = time.monotonic()
            return False

        self.wait_for(name + ': model readiness', check,
                      self.config.get('startup_timeout', 1200))

    def stop_nemor(self, drain=True):
        node = self.config['nodes'][self.nemor]
        for s in node['services'][:-1]:
            self.action(self.nemor, 'service', service=s, action='stop')
        if drain and self.journal['snapshots'][self.nemor]['services'][node['services'][-1]]:
            self.wait_for('Nemor idle', lambda: not self.call(self.nemor, 'busy')['busy'],
                          self.config.get('drain_timeout', 300))
        self.action(self.nemor, 'service', service=node['services'][-1], action='stop')

    def resume_nemor(self):
        node = self.config['nodes'][self.nemor]
        for s in reversed(node['services']):
            if self.journal['snapshots'][self.nemor]['services'][s]:
                self.action(self.nemor, 'service', service=s, action='start')
                if s == node['services'][-1]:
                    # busy() validates that both health and task APIs are serving.
                    self.wait_for('Nemor ready', lambda: isinstance(self.call(self.nemor, 'busy')['busy'], bool), 60)

    def release(self):
        for name in list(self.journal['locks']):
            self.action(name, 'unlock')
            self.journal['locks'].remove(name)
            self.save()

    def switch(self, mode):
        with self.exclusive():
            old = self.read_journal()
            if old and old['status'] not in ('complete', 'rolled_back', 'aborted'):
                raise LayoutError('Unfinished transaction; run recover before another switch')
            plan = self.plan(mode)  # Validation before locks or service changes.
            if plan['already_active']:
                self.output('Mode already active: ' + mode, flush=True)
                return
            self.journal = {'id': uuid.uuid4().hex, 'fingerprint': self.fingerprint,
                            'mode': mode, 'status': 'preparing', 'locks': [], 'snapshots': {},
                            'mutated': False, 'events': [], 'pending': None}
            self.save()
            try:
                for name in self.config['nodes']:
                    # Persist intent before the remote call, including a lost SSH acknowledgement.
                    self.journal['locks'].append(name)
                    self.save()
                    result = self.action(name, 'lock')
                    if not result.get('acquired'):
                        self.journal['locks'].remove(name)
                        self.save()
                        raise LayoutError(name + ': another transaction owns the host')
                for name in self.config['nodes']:
                    self.journal['snapshots'][name] = self.action(name, 'snapshot', modes=list(self.config['modes']))
                    self.save()
                self.journal.update(status='switching', mutated=True)
                self.save()
                self.stop_nemor()
                for name in self.proxies:
                    self.action(name, 'service', service=self.config['nodes'][name]['service'], action='stop')
                for name in self.config['nodes']:
                    self.action(name, 'configure', mode=mode, desired=self.config['modes'][mode].get(name, []))
                for name in self.proxies:
                    self.action(name, 'service', service=self.config['nodes'][name]['service'], action='start')
                for name in self.proxies:
                    self.wait_models(name, self.config['modes'][mode][name])
                self.resume_nemor()
                self.journal['status'] = 'committed'
                self.save()
                self.release()
                self.journal['status'] = 'complete'
                self.event('Mode active: ' + mode)
            except (Exception, KeyboardInterrupt) as exc:
                self.journal['error'] = str(exc) or type(exc).__name__
                self.save()
                # Do not undo a completed switch merely because an unlock acknowledgement was lost.
                if self.journal['status'] == 'committed':
                    raise LayoutError('Mode applied; lock cleanup incomplete. Run recover.') from exc
                try:
                    self._recover()
                except (Exception, KeyboardInterrupt) as rollback_error:
                    self.journal['status'] = 'recovery_required'
                    self.journal['recovery_error'] = str(rollback_error)
                    self.save()
                    raise LayoutError(f'Switch failed: {exc}; recovery failed: {rollback_error}. Run recover.') from exc
                raise LayoutError(f'Switch failed; previous state restored: {exc}') from exc

    def _recover(self):
        if self.journal['mutated']:
            self.journal['status'] = 'recovering'
            self.save()
            self.stop_nemor(drain=False)
            for name in self.proxies:
                self.action(name, 'service', service=self.config['nodes'][name]['service'], action='stop')
            # Stop every proxy successfully BEFORE restoring/starting either placement.
            for name in self.config['nodes']:
                self.action(name, 'restore')
            for name in self.proxies:
                node = self.config['nodes'][name]
                snapshot = self.journal['snapshots'][name]
                if snapshot['services'][node['service']]:
                    self.action(name, 'service', service=node['service'], action='start')
                    expected = [n for n, s in snapshot['runtimes'].items() if s['status'] == 'running']
                    autoload = [n for n in expected if snapshot['runtimes'][n]['autoload']]
                    self.wait_models(name, autoload)
                    for runtime in expected:
                        if runtime not in autoload:
                            self.action(name, 'runtime', name=runtime, action='start')
                    self.wait_models(name, expected)
            self.resume_nemor()
        self.release()
        self.journal['status'] = 'rolled_back' if self.journal['mutated'] else 'aborted'
        self.event('Recovery complete')

    def recover(self):
        self.check_coordinator()
        with self.exclusive():
            self.journal = self.read_journal()
            if not self.journal or self.journal['status'] in ('complete', 'rolled_back', 'aborted'):
                raise LayoutError('No unfinished transaction')
            if self.journal['fingerprint'] != self.fingerprint:
                raise LayoutError('Configuration changed; restore the configuration used for this transaction')
            # Reacquire only our transaction's locks. Never steal another coordinator's locks.
            for name in self.journal['locks']:
                result = self.call(name, 'lock', tx=self.journal['id'])
                if not result.get('acquired'):
                    raise LayoutError(name + ': another transaction owns the host')
            if self.journal['status'] == 'committed':
                self.release()
                self.journal['status'] = 'complete'
                self.save()
                return
            self._recover()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default=str(HERE / 'layouts.yaml'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    for verb in ('plan', 'switch'):
        p = sub.add_parser(verb)
        p.add_argument('mode')
        if verb == 'switch':
            p.add_argument('--dry-run', action='store_true')
    sub.add_parser('recover')
    args = parser.parse_args(argv)
    try:
        c = Controller(yaml.safe_load(Path(args.config).expanduser().read_text()))
        if args.command == 'status':
            print(json.dumps(c.status(), ensure_ascii=False, indent=2))
        elif args.command == 'plan' or getattr(args, 'dry_run', False):
            print(json.dumps(c.plan(args.mode), ensure_ascii=False, indent=2))
        elif args.command == 'switch':
            c.switch(args.mode)
        else:
            c.recover()
        return 0
    except (LayoutError, OSError, ValueError, KeyError) as exc:
        print('model-layout: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
