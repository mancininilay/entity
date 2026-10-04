#!/usr/bin/env python3
"""Write an Entity input file for the P1.3 `rec_mix` generator.

The physical inputs follow p1.1 (SM S1): sigma (total-mass magnetization),
q = n_i/n_e-, mu = m_i/m_e, d_up in cells. The Entity scales are derived as in
the P1.3 technical spec III.1:

    D          = (2 - q + q mu)/2          (rho/(n0 me), n0 = all macroparticles)
    sigma0     = sigma D  = sigma_e*
    skindepth0 = d_up sqrt((2 - q + q/mu)/2)
    larmor0    = skindepth0 / sqrt(sigma0)

For the test-particle layout q = 0, sigma0 = sigma and skindepth0 = d_up.
The generator refuses [scales] that disagree with these values.

Examples
--------
C2-a (active, standard domain):
    make_input.py --name c2a --layout active --sigma 40 --q 0.1 --mu 25
C2-d (passive):
    make_input.py --name c2d --layout testprtl --sigma 160 --test-masses 25 100 400
"""

import argparse
import math


def derived(layout, sigma, q, mu, dup):
    if layout == "active":
        D = (2.0 - q + q * mu) / 2.0
        sigma0 = sigma * D
        skin = dup * math.sqrt((2.0 - q + q / mu) / 2.0)
    else:
        D, sigma0, skin = 1.0, sigma, dup
    return D, sigma0, skin, skin / math.sqrt(sigma0)


def fmt(x):
    return repr(float(x))


def bytes_per_slot(npld_i=1, npld_r=0, tracking=False):
    """2D single precision: i1, i2, i1_prev, i2_prev (int32), dx1, dx2 and their
    previous values (float), ux1-3, weight (float), tag (int16), integer and
    real payloads. Sorting and communication buffers come on top."""
    ni = npld_i + (2 if tracking else 0)  # MPI tracking uses two slots
    return 4 * 4 + 4 * 4 + 3 * 4 + 4 + 2 + 4 * ni + 4 * npld_r


def capacities(a, act):
    """Per-species, per-domain capacities (Codex IC2). Entity allocates maxnpart
    for every species on every domain, so the estimate uses the largest local
    domain and the largest nominal per-cell density each species can have
    there: an upstream species fills its half box, the sheet's trial count
    (rejection sampling) covers the whole domain, tracers fill both halves."""
    if a.maxnpart is not None:
        caps = [float(a.maxnpart)] * 9
    else:
        n1d, n2d = a.decomposition
        if n1d <= 0 or n2d <= 0:
            raise SystemExit("make_input: give --decomposition N1D N2D (both > 0) for the capacity "
                             "estimate, or an explicit --maxnpart")
        cells = math.ceil(a.n1 / n1d) * math.ceil(a.n2 / n2d)
        h = 0.5 * a.ppc0
        q = a.q if act else 0.0
        eta = a.cs_overdensity
        if act:
            up = [h, h * (1 - q), h * q]          # e-, e+, ion of one side
            sheet = [h * eta, h * eta * (1 - q), h * eta * q]
        else:
            up = [h, h, h]                        # e-, e+ of one side; tracer: both sides
            sheet = [h * eta, h * eta, 0.0]
        caps = []
        for s in range(9):
            k = s % 3
            if s < 3 and not (not act and k == 2):
                n = a.sheet_reserve * sheet[k] * cells
            else:
                n = a.reserve * up[k] * cells
            caps.append(max(1000.0, math.ceil(n + 10 * math.sqrt(max(n, 1.0)))))
    bps = bytes_per_slot(tracking=a.tracking)
    gib = sum(caps) * bps / 2**30
    if gib > a.max_gib_per_domain:
        raise SystemExit(f"make_input: particle arrays need {gib:.1f} GiB per domain "
                         f"(> {a.max_gib_per_domain}); use more domains or a smaller reserve")
    return caps, gib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--layout", choices=["active", "testprtl"], default="active")
    ap.add_argument("--sheet-init", choices=["legacy", "balanced"], default="legacy")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--test-masses", type=float, nargs=3, default=[25.0, 100.0, 400.0])
    ap.add_argument("--test-temperatures", type=float, nargs=3, default=None,
                    help="physical kT of the tracers in me c^2 (default theta_up)")
    ap.add_argument("--n1", type=int, default=8320, help="outflow cells (x1)")
    ap.add_argument("--n2", type=int, default=3000, help="inflow cells (x2)")
    ap.add_argument("--dup", type=float, default=5.0, help="d_up in cells")
    ap.add_argument("--ppc0", type=float, default=32.0)
    ap.add_argument("--theta-up", type=float, default=0.01)
    ap.add_argument("--cs-width", type=float, default=400.0, help="a in cells")
    ap.add_argument("--cs-overdensity", type=float, default=3.0)
    ap.add_argument("--padding", type=int, default=12)
    ap.add_argument("--ds-abs", type=int, default=10)
    ap.add_argument("--filters", type=int, default=8)
    ap.add_argument("--cfl", type=float, default=0.45 * math.sqrt(2.0))
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--t-open-tlc", type=float, default=1.35)
    ap.add_argument("--usr-interval", type=int, default=500)
    ap.add_argument("--line-interval", type=int, default=100)
    ap.add_argument("--cv-interval", type=int, default=1)
    ap.add_argument("--field-interval", type=int, default=5000)
    ap.add_argument("--field-stride", type=int, default=8)
    ap.add_argument("--checkpoint-interval", type=int, default=20000)
    ap.add_argument("--maxnpart", type=float, default=None,
                    help="per-species, per-domain capacity for all species (overrides the estimate)")
    ap.add_argument("--decomposition", type=int, nargs=2, default=[-1, -1],
                    help="domains along x1 and x2; required (both > 0) for the capacity estimate")
    ap.add_argument("--reserve", type=float, default=3.0,
                    help="capacity reserve over the nominal per-domain load of upstream species "
                         "and tracers (compression, injection, migration)")
    ap.add_argument("--sheet-reserve", type=float, default=2.0,
                    help="capacity reserve over the sheet trial count (plasmoids carry sheet "
                         "particles between domains)")
    ap.add_argument("--pusher", default="Vay", help="particle pusher (continuity numerics: Vay, D3)")
    ap.add_argument("--max-gib-per-domain", type=float, default=30.0,
                    help="refuse capacities whose particle arrays exceed this per domain")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tracking", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    D, sigma0, skin, larmor = derived(a.layout, a.sigma, a.q, a.mu, a.dup)
    dt = a.cfl / math.sqrt(2.0)  # Entity: dt = CFL * dx / sqrt(D)
    runtime = a.steps * dt
    w_out = 2 * a.ds_abs
    w_in = 1.5 * (a.padding - a.filters)
    act = a.layout == "active"
    caps, gib = capacities(a, act)
    masses = [1.0, 1.0, a.mu] if act else None

    L = []
    L.append("# generated by pgens/rec_mix/make_input.py")
    L.append(f"# layout={a.layout} sigma={a.sigma} q={a.q} mu={a.mu} D={D:.6g} sigma0={sigma0:.9g}")
    L.append("[simulation]")
    L.append(f'  name    = "{a.name}"')
    L.append('  engine  = "srpic"')
    L.append(f"  runtime = {fmt(runtime)}")
    if a.seed is not None:
        L.append(f"  seed    = {a.seed}")
    L.append("")
    L.append("  [simulation.domain]")
    L.append(f"    decomposition = [{a.decomposition[0]}, {a.decomposition[1]}]")
    L.append("")
    L.append("[grid]")
    L.append(f"  resolution = [{a.n1}, {a.n2}]")
    L.append(f"  extent     = [[0.0, {fmt(a.n1)}], [0.0, {fmt(a.n2)}]]")
    L.append("")
    L.append("  [grid.metric]")
    L.append('    metric = "minkowski"')
    L.append("")
    L.append("  [grid.boundaries]")
    L.append('    fields    = [["PERIODIC"], ["MATCH", "MATCH"]]')
    L.append('    particles = [["PERIODIC"], ["ABSORB", "ABSORB"]]')
    L.append("")
    L.append("    [grid.boundaries.match]")
    L.append(f"      ds = [[{fmt(w_out)}], [{fmt(w_in)}]]")
    L.append("")
    L.append("[scales]")
    L.append(f"  larmor0    = {larmor!r}")
    L.append(f"  skindepth0 = {skin!r}")
    L.append("")
    L.append("[algorithms]")
    L.append(f"  current_filters = {a.filters}")
    L.append("")
    L.append("  [algorithms.timestep]")
    L.append(f"    CFL = {a.cfl!r}")
    L.append("")
    L.append("[particles]")
    L.append(f"  ppc0 = {fmt(a.ppc0)}")
    labels = ["e-_cs", "e+_cs", "i_cs", "e-_lo", "e+_lo", "i_lo", "e-_up", "e+_up", "i_up"]
    if not act:
        labels = ["e-_cs", "e+_cs", "tr_a", "e-_lo", "e+_lo", "tr_b", "e-_up", "e+_up", "tr_c"]
    for s in range(9):
        lep = s % 3 != 2
        if act:
            m = 1.0 if lep else a.mu
        else:
            m = 1.0 if lep else a.test_masses[s // 3]
        ch = (-1.0 if s % 3 == 0 else 1.0) if lep else 1.0
        L.append("")
        L.append("  [[particles.species]]")
        L.append(f'    label          = "{labels[s]}"')
        L.append(f"    mass           = {fmt(m)}")
        L.append(f"    charge         = {fmt(ch)}")
        L.append(f"    maxnpart       = {float(caps[s]):.6e}")
        L.append(f'    pusher         = "{a.pusher}"')
        L.append("    n_payloads_int = 1")
        if not act and not lep:
            L.append("    deposit        = false")
        if a.tracking:
            L.append("    tracking       = true")
    L.append("")
    L.append("[setup]")
    L.append(f'  layout           = "{a.layout}"')
    L.append(f'  sheet_init       = "{a.sheet_init}"')
    L.append(f"  sigma            = {fmt(a.sigma)}")
    if act:
        L.append(f"  q                = {fmt(a.q)}")
        L.append(f"  mu               = {fmt(a.mu)}")
    else:
        L.append(f"  test_masses      = [{', '.join(fmt(x) for x in a.test_masses)}]")
        if a.test_temperatures is not None:
            L.append(f"  test_temperatures = [{', '.join(fmt(x) for x in a.test_temperatures)}]")
    L.append(f"  dup_cells        = {fmt(a.dup)}")
    L.append(f"  theta_up         = {fmt(a.theta_up)}")
    L.append(f"  cs_width         = {fmt(a.cs_width)}")
    L.append(f"  cs_overdensity   = {fmt(a.cs_overdensity)}")
    L.append(f"  t_open_tlc       = {fmt(a.t_open_tlc)}")
    L.append(f"  injector_padding = {a.padding}")
    L.append(f"  ds_abs           = {a.ds_abs}")
    L.append("  mixing_threshold = 0.01")
    L.append(f"  usr_interval     = {a.usr_interval}")
    L.append(f"  line_interval    = {a.line_interval}")
    L.append(f"  cv_interval      = {a.cv_interval}")
    L.append("  cv_margin        = 10")
    L.append("  spec_min         = 0.001")
    L.append("  spec_max         = 100000.0")
    L.append("  spec_num         = 300")
    L.append("  bulk_window      = 2")
    L.append("")
    L.append("[output]")
    L.append('  format = "BPFile"')
    L.append("")
    L.append("  [output.fields]")
    L.append(f"    interval     = {a.field_interval}")
    L.append(f"    downsampling = {a.field_stride}")
    # per-species mean 3-velocity V_s (Nilay 10-04) and, for continuity with
    # p1.1, the mass-weighted velocity of all depositing species (Tristan velx)
    dep = list(range(1, 10)) if act else [1, 2, 4, 5, 7, 8]
    flds = ['"E"', '"B"', '"J"', '"N_4_5_6"' if act else '"N_4_5"', '"N_7_8_9"' if act else '"N_7_8"']
    flds += [f'"N_{s}"' for s in range(1, 10)]
    flds += [f'"V_{s}"' for s in range(1, 10)]
    flds += ['"V_' + "_".join(str(s) for s in dep) + '"']
    L.append(f"    quantities   = [{', '.join(flds)}]")
    L.append('    custom       = ["mixing", "mask"]')
    L.append("")
    L.append("    # Tristan-like top-hat smoothing over 5x5 cells (output_dens_smooth = 2)")
    L.append("    [output.fields.smoothing]")
    L.append("      order  = 4")
    L.append('      method = "const"')
    L.append("")
    L.append("  [output.particles]")
    L.append("    enable = false")
    L.append("")
    L.append("  [output.spectra]")
    L.append("    enable = false")
    L.append("")
    L.append("  [output.stats]")
    L.append("    enable   = true")
    L.append("    interval = 100")
    L.append("")
    L.append("[checkpoint]")
    L.append(f"  interval = {a.checkpoint_interval}")
    L.append("  keep     = 2")
    out = a.out or f"{a.name}.toml"
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}: sigma0={sigma0:.9g} skindepth0={skin:.9g} larmor0={larmor:.9g} "
          f"dt={dt:.6g} runtime={runtime:.6g} t_lc={a.n1} w_out={w_out} w_in={w_in} "
          f"pusher={a.pusher} capacity={caps[3]:.3g} (e- side species) "
          f"particle arrays {gib:.2f} GiB per domain")


if __name__ == "__main__":
    main()
