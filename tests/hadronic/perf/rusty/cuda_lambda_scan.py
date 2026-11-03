#!/usr/bin/env python3
"""Static scan of translation units for the NVCC restrictions on extended lambdas and device code (J1, job 7386833).

The Mac cannot compile CUDA; this scan lists, file by file and line by line, every place where NVCC (CUDA 12.x,
`--expt-extended-lambda`, no `--expt-relaxed-constexpr`) refuses or warns about code that a host compiler accepts, so
that a failure of the compile-only job points to a file and line already listed. It is a structural heuristic (comments
and strings stripped, braces matched), not a C++ parser: every flagged item is read by hand and dispositioned in the
report; an empty flag list is evidence, not proof (the proof is the compile-only job).

Checks on every extended lambda (`Lambda`, `ClassLambda`, `KOKKOS_LAMBDA`, `KOKKOS_CLASS_LAMBDA`):
  L1  `if constexpr` inside the lambda body (a variable first captured there is refused: J1)
  L2  `auto` parameter (generic extended lambdas are refused)
  L3  an extended lambda nested inside another extended lambda
  L4  enclosing function is a constructor or destructor (its address cannot be taken)
  L5  enclosing function has a deduced return type (`auto f(...)` / `decltype(auto)` without a trailing type)
  L6  enclosing function is a private or protected member, or sits in a function-local class
  L7  enclosing host lambda is generic (`auto` parameter)
  L8  the body uses a structured binding of the enclosing function, or a host container / host-only object declared
      there (std::vector, std::string, std::map, std::function, std::shared_ptr, std::unique_ptr, std::ostream ...)
  L9  `this` in the body, or `ClassLambda` (captures *this by value: the class must be device-copyable)
Checks on device-callable code (extended lambda bodies and `Inline` / `KOKKOS_INLINE_FUNCTION` / `KOKKOS_FUNCTION`
function bodies, including functor `operator()`):
  D1  `throw` or a host `raise::` (raise::KernelError and KernelNotImplementedError are Inline, device-safe)
  D2  non-const `static` local variable
  D3  host-only library use (std::vector, std::string, std::map, std::function, std::cout, std::to_string, new/delete)
  D4  calls to constexpr host functions that NVCC warns about without --expt-relaxed-constexpr (std::numeric_limits,
      std::min, std::max, std::clamp): warning #20013, compiled and folded in job 7386833; listed, not changed
Checks on kernel launches (`Kokkos::parallel_for/reduce/scan`):
  K1  a function-local type (struct/class defined inside a function body) used in a launch or a lambda body: refused as
      a template argument of a __global__ function
Usage: cuda_lambda_scan.py <file> [<file> ...]  (prints one block per file; exit 0)
"""
import re
import sys

EXT = re.compile(r'\b(Lambda|ClassLambda|KOKKOS_LAMBDA|KOKKOS_CLASS_LAMBDA)\b')
DEVFN = re.compile(r'\b(Inline|KOKKOS_INLINE_FUNCTION|KOKKOS_FUNCTION|KOKKOS_FORCEINLINE_FUNCTION)\b')
LAUNCH = re.compile(r'Kokkos::parallel_(for|reduce|scan)\b')
HOSTTYPES = r'std::(vector|string|map|unordered_map|set|function|shared_ptr|unique_ptr|ostream|ofstream|ifstream|stringstream|array<[^>]*std::string)'
CTRL = {'if', 'constexpr', 'for', 'while', 'switch', 'catch', 'return', 'sizeof', 'decltype', 'alignof', 'static_assert', 'else', 'do'}


def strip(src):
  """comments and string/char literals replaced by spaces (newlines kept, so offsets and lines are preserved)"""
  out, i, n = [], 0, len(src)
  while i < n:
    c = src[i]
    if src.startswith('//', i):
      j = src.find('\n', i)
      j = n if j < 0 else j
      out.append(' ' * (j - i)); i = j
    elif src.startswith('/*', i):
      j = src.find('*/', i + 2)
      j = n if j < 0 else j + 2
      out.append(re.sub(r'[^\n]', ' ', src[i:j])); i = j
    elif c == '"' or (c == "'" and not (i > 0 and (src[i - 1].isalnum()))):
      q, j = c, i + 1
      while j < n and src[j] != q:
        j += 2 if src[j] == '\\' else 1
      j = min(j + 1, n)
      out.append(q + re.sub(r'[^\n]', ' ', src[i + 1:j - 1]) + q if j - i >= 2 else src[i:j]); i = j
    elif src.startswith('R"', i):
      m = re.match(r'R"([^(]*)\(', src[i:])
      end = src.find(')' + (m.group(1) if m else '') + '"', i)
      j = n if end < 0 else end + len(m.group(1) if m else '') + 2
      out.append(re.sub(r'[^\n]', ' ', src[i:j])); i = j
    else:
      out.append(c); i += 1
  return ''.join(out)


def match_brace(s, i):
  """index of the brace closing the one at s[i] ('{', '(' or '[')"""
  o = s[i]; c = {'{': '}', '(': ')', '[': ']'}[o]; d = 0
  for j in range(i, len(s)):
    if s[j] == o:
      d += 1
    elif s[j] == c:
      d -= 1
      if d == 0:
        return j
  return len(s) - 1


def line_of(s, i):
  return s.count('\n', 0, i) + 1


def blocks(s):
  """every '{...}' block with what introduces it: (open, close, header text before the brace)"""
  out, stack = [], []
  for i, ch in enumerate(s):
    if ch == '{':
      stack.append(i)
    elif ch == '}' and stack:
      o = stack.pop()
      k = o - 1
      # header: back to the previous ';', '{' or '}' at the same nesting (bounded)
      depth = 0
      while k >= 0:
        if s[k] in ')]':
          depth += 1
        elif s[k] in '([':
          depth -= 1
        elif depth == 0 and s[k] in ';{}':
          break
        k -= 1
      out.append((o, i, s[k + 1:o]))
  return out


def classify(header):
  """kind of block from its header: function, lambda, class, namespace, control, init, other"""
  h = ' '.join(header.split())
  if re.search(r'\]\s*(\([^)]*\))?\s*(mutable\s*)?(->\s*[^{]+)?$', h) and re.search(r'\[[^\]]*\]', h):
    return 'lambda', h
  if re.search(r'\b(struct|class|union)\s+\w+[^(]*$', h) or re.search(r'\b(struct|class)\s*$', h):
    return 'class', h
  if re.search(r'\bnamespace\b', h) or h.startswith('extern "C"'):
    return 'namespace', h
  if re.search(r'\b(enum)\b', h):
    return 'other', h
  m = re.search(r'([~\w:]+)\s*\([^()]*(\([^()]*\)[^()]*)*\)\s*(const)?\s*(noexcept)?\s*(override)?\s*(->\s*[^{]+)?\s*(:\s*[^{]*)?$', h)
  if m:
    name = m.group(1).split('::')[-1]
    if name in CTRL:
      return 'control', h
    return 'function', h
  if re.search(r'\b(else|do|try)\s*$', h) or h == '':
    return 'control', h
  return 'other', h


def enclosing(blks, pos):
  """the blocks containing pos, innermost first"""
  return sorted([b for b in blks if b[0] < pos < b[1]], key=lambda b: -b[0])


def fn_name(h):
  m = re.search(r'([~\w:]+)\s*\(', h)
  return m.group(1) if m else '?'


def scan(path):
  raw = open(path, errors='ignore').read()
  s = strip(raw)
  blks = blocks(s)
  kinds = {b[0]: classify(b[2]) for b in blks}
  flags, sites = [], []

  def access_of(cls_block, pos):
    o, c, h = cls_block
    kw = re.search(r'\b(struct|class)\s+(\w+)\s*(final\s*)?(:[^{]*)?$', ' '.join(h.split()))
    default = 'private' if kw and kw.group(1) == 'class' else 'public'
    last = default
    for m in re.finditer(r'\b(public|private|protected)\s*:', s[o:pos]):
      # only specifiers at the class's own nesting level
      inner = s[o + 1:o + m.start()]
      if inner.count('{') == inner.count('}'):
        last = m.group(1)
    return last

  # function-local types (K1)
  local_types = []
  for o, c, h in blks:
    k, hh = kinds[o]
    if k == 'class':
      outer = enclosing(blks, o)
      if any(kinds[b[0]][0] in ('function', 'lambda') for b in outer):
        m = re.search(r'\b(struct|class|union)\s+(\w+)', hh)
        if m:
          local_types.append((m.group(2), line_of(s, o)))

  # extended lambdas
  for m in EXT.finditer(s):
    p = m.end()
    while p < len(s) and s[p] in ' \t\n':
      p += 1
    params = ''
    if p < len(s) and s[p] == '(':
      q = match_brace(s, p); params = s[p + 1:q]; p = q + 1
    b = s.find('{', p)
    if b < 0:
      continue
    e = match_brace(s, b)
    body = s[b:e + 1]
    ln = line_of(s, m.start())
    outer = enclosing(blks, m.start())
    fn, host_lambdas, cls = None, [], None
    for blk in outer:
      k, hh = kinds[blk[0]]
      if k == 'lambda' and fn is None:
        host_lambdas.append((blk, hh))
      elif k == 'function' and fn is None:
        fn = (blk, hh)
      elif k == 'class' and fn is not None and cls is None:
        cls = (blk, hh)
    name = fn_name(fn[1]) if fn else '?'
    site = {'line': ln, 'macro': m.group(1), 'fn': name, 'flags': []}
    if 'if constexpr' in body:
      site['flags'].append('L1 if constexpr in the body')
    if re.search(r'\bauto\b', params):
      site['flags'].append('L2 auto parameter')
    if EXT.search(body[1:]):
      site['flags'].append('L3 nested extended lambda')
    if fn:
      h = ' '.join(fn[1].split())
      short = name.split('::')[-1]
      clsname = None
      if cls:
        mm = re.search(r'\b(struct|class)\s+(\w+)\s*(final\s*)?(:[^{]*)?$', ' '.join(cls[1].split()))
        clsname = mm.group(2) if mm else None
      if short.startswith('~') or (clsname and short == clsname) or (len(name.split('::')) >= 2 and name.split('::')[-1] == name.split('::')[-2]):
        site['flags'].append('L4 enclosing function is a constructor/destructor')
      if re.match(r'^(template\s*<[^{]*>\s*)?((static|inline|constexpr|virtual)\s+)*(auto|decltype\(auto\))\s+[\w:~]+\s*\(', h) and '->' not in h:
        site['flags'].append('L5 enclosing function has a deduced return type')
      if cls:
        acc = access_of(cls[0], m.start())
        if acc != 'public':
          site['flags'].append(f'L6 enclosing member function is {acc}')
        if any(kinds[b[0]][0] in ('function', 'lambda') for b in enclosing(blks, cls[0][0])):
          site['flags'].append('L6 enclosing class is function-local')
      fbody = s[fn[0][0]:m.start()]
      for sb in re.finditer(r'\bauto\s*&{0,2}\s*\[([^\]]+)\]\s*=', fbody):
        for nm in [x.strip() for x in sb.group(1).split(',')]:
          if re.search(r'\b' + re.escape(nm) + r'\b', body):
            site['flags'].append(f'L8 uses structured binding "{nm}"')
      for hv in re.finditer(HOSTTYPES + r'[^;(){}]*?\s(\w+)\s*(\{|=|;|\()', fbody):
        nm = hv.group(hv.lastindex - 1)
        if nm and re.search(r'\b' + re.escape(nm) + r'\b', body):
          site['flags'].append(f'L8 uses host object "{nm}" ({hv.group(1)})')
    for hb, hh in host_lambdas:
      pm = re.search(r'\]\s*\(([^)]*)\)', ' '.join(hh.split()))
      if pm and re.search(r'\bauto\b', pm.group(1)):
        site['flags'].append('L7 inside a generic host lambda')
    if re.search(r'\bthis\b', body) or m.group(1) in ('ClassLambda', 'KOKKOS_CLASS_LAMBDA'):
      site['flags'].append('L9 this / *this capture')
    site['flags'] += [f for f in device_flags(body)]
    for t, tl in local_types:
      if re.search(r'\b' + t + r'\b', body):
        site['flags'].append(f'K1 local type {t} (line {tl}) used in the body')
    sites.append(site)

  # device-callable functions
  devfns = []
  for m in DEVFN.finditer(s):
    b = s.find('{', m.end())
    semi = s.find(';', m.end())
    if b < 0 or (0 <= semi < b):
      continue
    hdr = ' '.join(s[m.end():b].split())
    e = match_brace(s, b)
    fl = device_flags(s[b:e + 1])
    devfns.append({'line': line_of(s, m.start()), 'fn': fn_name(hdr), 'flags': fl})

  # launches
  launches = []
  for m in LAUNCH.finditer(s):
    q = s.find('(', m.end())
    e = match_brace(s, q)
    args = s[q:e + 1]
    lf = []
    for t, tl in local_types:
      if re.search(r'\b' + t + r'\b', args):
        lf.append(f'K1 local type {t} (line {tl}) in the launch')
    launches.append({'line': line_of(s, m.start()), 'kind': m.group(1), 'lambda': bool(EXT.search(args)),
                     'args': ' '.join(args.split())[:110], 'flags': lf})
  return sites, devfns, launches, local_types


def device_flags(body):
  fl = []
  if re.search(r'\bthrow\b', body):
    fl.append('D1 throw')
  host_raise = sorted(set(r for r in re.findall(r'\braise::(\w+)', body) if not r.startswith('Kernel')))
  if host_raise:
    fl.append('D1 host raise::' + ', raise::'.join(host_raise))
  for st in re.finditer(r'\bstatic\s+(?!constexpr|const\b|_assert|cast)[\w:<>]+\s+\w+', body):
    fl.append('D2 static local: ' + ' '.join(st.group(0).split()))
  if re.search(HOSTTYPES + r'|std::cout|std::cerr|std::to_string|\bnew\s+\w|\bdelete\b', body):
    fl.append('D3 host-only library use')
  d4 = sorted(set(re.findall(r'std::(numeric_limits|min|max|clamp)\b', body)))
  if d4:
    fl.append('D4 constexpr host call std::' + ', std::'.join(d4))
  return fl


def main():
  for path in sys.argv[1:]:
    sites, devfns, launches, local_types = scan(path)
    nflag = sum(1 for x in sites if x['flags']) + sum(1 for x in devfns if x['flags']) + sum(1 for x in launches if x['flags'])
    print(f'=== {path}: {len(sites)} extended lambdas, {len(devfns)} device-callable functions, {len(launches)} launches, '
          f'{len(local_types)} function-local types; {nflag} flagged')
    for x in sites:
      print(f'  lambda   :{x["line"]:<5} {x["macro"]:<11} in {x["fn"]:<28} {"; ".join(x["flags"]) or "clean"}')
    for x in devfns:
      if x['flags']:
        print(f'  device fn:{x["line"]:<5} {x["fn"]:<40} {"; ".join(x["flags"])}')
    for x in launches:
      print(f'  launch   :{x["line"]:<5} parallel_{x["kind"]:<7} {"lambda " if x["lambda"] else "functor"} {"; ".join(x["flags"]) or "clean"}  {x["args"][:80]}')
    for t, tl in local_types:
      print(f'  local type:{tl:<5} {t}')


if __name__ == '__main__':
  main()
