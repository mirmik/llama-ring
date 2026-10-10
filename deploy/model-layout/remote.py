"""Ephemeral SSH worker. No daemon or llm_proxy imports; JSON on stdin/stdout."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


def atomic(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def http(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    # Control and readiness checks are always host-local, never through an env proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
        return json.load(r)


def systemctl(action, service):
    p = subprocess.run(['systemctl', '--user', action, service], capture_output=True, text=True, timeout=90)
    if p.returncode:
        raise RuntimeError(f'systemctl {action} {service} failed ({p.returncode})')


def active(service):
    p = subprocess.run(['systemctl', '--user', 'show', service, '-p', 'LoadState', '-p', 'ActiveState'],
                       capture_output=True, text=True, timeout=10, check=True)
    props = dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)
    if props.get('LoadState') != 'loaded':
        raise RuntimeError(f'Service is not loaded: {service}')
    state = props.get('ActiveState')
    if state not in ('active', 'inactive', 'failed'):
        raise RuntimeError(f'Service is transitioning: {service}: {state}')
    return state == 'active'


def read_config(node):
    import yaml
    return yaml.safe_load(Path(node['config']).expanduser().read_text())


def runtime_states(node):
    return {r['name']: r for r in http(node['control_url'] + '/v1/control/state')['runtimes']}


def ready(runtime):
    if runtime['status'] != 'running' or not runtime.get('health_url'):
        return False
    try:
        http(runtime['health_url'], timeout=3)
        return True
    except (OSError, ValueError):
        return False


def busy(node):
    h = http(node['health_url'])
    if 'processing_sessions' not in h:
        raise RuntimeError('Nemor health must report processing_sessions')
    tasks = http(node['tasks_url'])
    if not isinstance(tasks.get('tasks'), list):
        raise RuntimeError('Unsupported Nemor tasks response')
    return bool(h['processing_sessions'] or any(
        t.get('status') not in ('completed', 'failed', 'cancelled', 'canceled', 'done')
        for t in tasks['tasks']))


def inspect(node):
    services = node.get('services', [node.get('service')])
    result = {'services': {s: active(s) for s in services}}
    if node['kind'] == 'proxy':
        c = read_config(node)
        states = runtime_states(node) if result['services'][node['service']] else {}
        result['runtimes'] = {n: {'status': r['status'], 'ready': ready(r),
                                 'autoload': r.get('autoload'), 'on_demand': r.get('on_demand')}
                              for n, r in states.items()}
        result['configured'] = sorted(c.get('runtimes', {}))
        result['inference_settings'] = {
            k: c.get('inference', {}).get(k)
            for settings in node.get('inference_settings', {}).values() for k in settings}
        result['routes'] = {n: {'kind': r['kind'], 'health_path': r.get('health_path', '/health'), 'backends': [
            {**{k: b[k] for k in ('url', 'model', 'auth') if k in b},
             'tls_verify': b.get('tls_verify', b.get('verify_tls', True))} for b in r.get('backends', [])]}
            for n, r in c.get('routes', {}).items()}
    else:
        base = Path(node['directory']).expanduser()
        result['characters'] = {}
        result['character_settings'] = {}
        for name in node['characters']:
            c = json.loads((base / 'characters' / (name + '.json')).read_text())
            result['characters'][name] = c.get('llm_profile')
            result['character_settings'][name] = {
                k: c.get(k) for settings in node.get('character_settings', {}).values()
                for k in settings.get(name, {})}
        result['busy'] = busy(node) if result['services'][node['services'][-1]] else False
    return result


def check_backend(backend, host):
    import ssl
    headers = {'Authorization': 'Bearer ' + host['token']}
    if host.get('host_id'):
        headers['X-LLM-Proxy-Host-ID'] = host['host_id']
    handlers = [urllib.request.ProxyHandler({})]
    if backend.get('tls_verify') is False:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    request = urllib.request.Request(backend['url'].rstrip('/') + '/v1/models', headers=headers)
    with urllib.request.build_opener(*handlers).open(request, timeout=10) as response:
        json.load(response)


def diagnostics(node, names):
    from urllib.parse import quote
    states = runtime_states(node)
    result = {}
    for name in names:
        runtime = states.get(name)
        if runtime is None:
            result[name] = {'ready': False, 'failed': True, 'status': 'missing'}
            continue
        is_ready = ready(runtime)
        failed = False
        if runtime['status'] == 'stopped':
            logs = http(node['control_url'] + '/v1/control/runtimes/' + quote(name, safe='') + '/logs').get('logs', [])
            failed = any(line.startswith('$ ') or 'Autostart failed:' in line for line in logs)
        result[name] = {'ready': is_ready, 'failed': failed,
                        'status': 'ready' if is_ready else ('exited' if failed else
                                  'loading' if runtime['status'] == 'running' else 'queued')}
    return result


def preflight(node, modes):
    result = inspect(node)
    if node['kind'] == 'proxy':
        c = read_config(node)
        runtimes = dict(c.get('runtimes', {}))
        runtimes.update({k: v for k, v in node.get('additions', {}).items() if k not in runtimes})
        missing = set(node['managed']) - runtimes.keys()
        if missing:
            raise RuntimeError('Missing runtime definitions: ' + ', '.join(sorted(missing)))
        unmanaged = [n for n, r in c.get('runtimes', {}).items() if n not in node['managed'] and (
            r.get('autoload') or r.get('on_demand') or result['runtimes'].get(n, {}).get('status') == 'running')]
        if unmanaged:
            raise RuntimeError('Unmanaged runtimes could be affected: ' + ', '.join(unmanaged))
        if not result['services'][node['service']]:
            raise RuntimeError('Proxy must be running before a new switch')
        for n, r in c.get('runtimes', {}).items():
            if r.get('autoload') and result['runtimes'].get(n, {}).get('status') != 'running':
                raise RuntimeError('Autoload runtime is stopped; resolve before switching: ' + n)
        for filename in node.get('required_paths', []):
            if not Path(filename).expanduser().exists():
                raise RuntimeError('Missing required file: ' + filename)
        for check in node.get('python_checks', []):
            env = os.environ.copy()
            env.update(check.get('env', {}))
            checked = subprocess.run([str(Path(check['python']).expanduser()), '-c',
                                     'import importlib; ' + '; '.join('importlib.import_module(' + repr(m) + ')' for m in check['modules'])],
                                    env=env, capture_output=True, timeout=45)
            if checked.returncode:
                raise RuntimeError('Python dependency check failed: ' + check['python'])
        for mode in modes:
            for route in node.get('routes', {}).get(mode, {}).values():
                for b in route['backends']:
                    if b.get('auth'):
                        host = c.get('hosts', {}).get(b['auth'], {})
                        if not host.get('token'):
                            raise RuntimeError('Missing proxy host credential: ' + b['auth'])
                        check_backend(b, host)
        for name in node['managed']:
            r = runtimes[name]
            # Detect ports already used by unrelated runtimes before taking anything down.
            for other, spec in runtimes.items():
                if other not in node['managed'] and r.get('port') == spec.get('port') and r.get('port'):
                    raise RuntimeError(f'Port collision: {name} / {other}')
    else:
        base = Path(node['directory']).expanduser()
        for profiles in node['characters'].values():
            for mode in modes:
                p = base / 'model_profiles' / (profiles[mode] + '.json')
                c = json.loads(p.read_text())
                if not c.get('server') or not c.get('model'):
                    raise RuntimeError('Invalid Nemor profile: ' + p.name)
                expected = node.get('expected_profiles', {}).get(profiles[mode], {})
                if any(c.get(k) != v for k, v in expected.items()):
                    raise RuntimeError('Nemor profile points to an unexpected model/server: ' + p.name)
    return result


def root(node):
    return Path(node.get('state_dir', '~/.local/state/model-layout')).expanduser()


def lock(node, tx):
    directory = root(node)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / 'lock'
    try:
        # Exclusive creation writes the owner through the already-open fd.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_text() == tx:
            return True
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(tx)
        f.flush()
        os.fsync(f.fileno())
    return True


def owned(node, tx):
    path = root(node) / 'lock'
    if not path.exists() or path.read_text() != tx:
        raise RuntimeError('Remote lock not owned by this transaction')


def backup_dir(node, tx):
    if not re.fullmatch(r'[a-zA-Z0-9-]+', tx):
        raise ValueError('Invalid transaction ID')
    return root(node) / tx


def snapshot(node, tx):
    directory = backup_dir(node, tx)
    manifest = directory / 'snapshot.json'
    if manifest.exists():
        return json.loads(manifest.read_text())['state']
    state = inspect(node)
    paths = [Path(node['config']).expanduser()] if node['kind'] == 'proxy' else [
        Path(node['directory']).expanduser() / 'characters' / (name + '.json') for name in node['characters']]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = []
    for i, p in enumerate(paths):
        data = p.read_bytes()
        atomic(directory / str(i), data)
        files.append({'path': str(p), 'backup': str(i), 'sha256': hashlib.sha256(data).hexdigest()})
    atomic(manifest, json.dumps({'files': files, 'state': state}).encode())
    return state


def metadata(node, tx):
    return json.loads((backup_dir(node, tx) / 'snapshot.json').read_text())


def write_owned(node, tx, path, data):
    manifest = metadata(node, tx)
    for entry in manifest['files']:
        if entry['path'] == str(Path(path).expanduser()):
            entry['written_sha256'] = hashlib.sha256(data).hexdigest()
    atomic(backup_dir(node, tx) / 'snapshot.json', json.dumps(manifest).encode())
    atomic(path, data)


def configure(node, tx, mode, desired):
    # Call only with owned services stopped. Avoid silently overwriting edits since snapshot.
    for s in node.get('services', [node.get('service')]):
        if active(s):
            raise RuntimeError('Service still active: ' + s)
    for f in metadata(node, tx)['files']:
        data = Path(f['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != f['sha256']:
            raise RuntimeError('Configuration changed after snapshot: ' + f['path'])
    if node['kind'] == 'proxy':
        import yaml
        c = read_config(node)
        for n, spec in node.get('additions', {}).items():
            c.setdefault('runtimes', {}).setdefault(n, spec)
        for n, settings in node.get('overrides', {}).items():
            if n not in node['managed']:
                raise ValueError('Override outside managed runtimes: ' + n)
            c['runtimes'][n].update(settings)
        for n in node['managed']:
            c['runtimes'][n]['autoload'] = n in desired
            c['runtimes'][n]['on_demand'] = False
        for route in node.get('removed_routes', []):
            c.setdefault('routes', {}).pop(route, None)
        c.setdefault('routes', {}).update(node.get('routes', {}).get(mode, {}))
        patch_settings(c.setdefault('inference', {}), node.get('inference_settings', {}).get(mode, {}))
        write_owned(node, tx, Path(node['config']).expanduser(), yaml.safe_dump(c, allow_unicode=True, sort_keys=False).encode())
    else:
        for name, profiles in node['characters'].items():
            p = Path(node['directory']).expanduser() / 'characters' / (name + '.json')
            c = json.loads(p.read_text())
            c['llm_profile'] = profiles[mode]
            patch_settings(c, node.get('character_settings', {}).get(mode, {}).get(name, {}))
            write_owned(node, tx, p, (json.dumps(c, ensure_ascii=False, indent=2) + '\n').encode())


def patch_settings(config, settings):
    # null restores the application's default instead of writing an invalid null value.
    for key, value in settings.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value


def restore(node, tx):
    for s in node.get('services', [node.get('service')]):
        if active(s):
            raise RuntimeError('Refusing to restore while service is active: ' + s)
    for f in metadata(node, tx)['files']:
        digest = hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()
        if digest not in (f['sha256'], f.get('written_sha256')):
            raise RuntimeError('Concurrent edit; refusing to overwrite: ' + f['path'])
    for f in metadata(node, tx)['files']:
        atomic(f['path'], (backup_dir(node, tx) / f['backup']).read_bytes())


def dispatch(p):
    node, op = p['node'], p['op']
    if op == 'inspect':
        return inspect(node)
    if op == 'preflight':
        return preflight(node, p['modes'])
    if op == 'busy':
        return {'busy': busy(node)}
    if op == 'diagnostics':
        return diagnostics(node, p['names'])
    if op == 'health':
        states = runtime_states(node)
        return {n: ready(states[n]) if n in states else False for n in p['names']}
    tx = p['tx']
    if op == 'lock':
        return {'acquired': lock(node, tx)}
    if op == 'unlock' and not (root(node) / 'lock').exists():
        return {}
    owned(node, tx)
    if op == 'unlock':
        (root(node) / 'lock').unlink()
    elif op == 'snapshot':
        preflight(node, p['modes'])
        return snapshot(node, tx)
    elif op == 'service':
        allowed = node.get('services', [node.get('service')])
        if p['service'] not in allowed or p['action'] not in ('start', 'stop'):
            raise ValueError('Unsupported service action')
        systemctl(p['action'], p['service'])
    elif op == 'configure':
        configure(node, tx, p['mode'], p.get('desired', []))
    elif op == 'restore':
        restore(node, tx)
    elif op == 'runtime':
        if p['name'] not in node['managed'] or p['action'] not in ('start', 'stop'):
            raise ValueError('Unsupported runtime action')
        from urllib.parse import quote
        http(node['control_url'] + '/v1/control/runtimes/' + quote(p['name'], safe='') + '/' + p['action'], {}, 1200)
    else:
        raise ValueError('Unknown operation: ' + op)
    return {}


if __name__ == '__main__':
    try:
        print(json.dumps({'ok': True, 'result': dispatch(json.load(sys.stdin))}))
    except Exception as exc:
        # HTTP errors include status, not response body/configuration/credentials.
        print(json.dumps({'ok': False, 'error': str(exc)}))
        sys.exit(1)
