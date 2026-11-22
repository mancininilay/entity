# The runs of M1, M2 and M3 (PREREGISTRATION.md of this folder). Sourced by a
# driver that defines
#   run <group> <name> <precision> <ranks> <decomposition> key=value ...
# which writes the input with make_input.py into <root>/<group>/<precision>/
# r<ranks>/<name>/ and runs entity.xc there on <ranks> ranks.
# M1_N_IONS exists only for the small local runs on the Mac (default: the
# registered 8,000,000).

DECOMPOSITIONS=("1:" "2:2, 1" "4:2, 2")

m1_cases() { # precision
  local prec="$1" cfg ranks dec pol
  local common="order=ions_first fields=false neutralizer=false gauss=false isotropic=true n_ions=${M1_N_IONS:-8000000} mass_ratio=1836.15 maxnpart_leptons=1000000 sort_interval=0 diag_interval=1 rank_lines=true"
  for cfg in "${DECOMPOSITIONS[@]}"; do
    ranks="${cfg%%:*}"; dec="${cfg#*:}"
    for pol in on off; do
      # H: the 0.5 bath at Gamma_p = 6 (the tables of V2); L: the 1.8 bath at
      # Gamma_p = 1.2, the point of test A1b unchanged (tables of the reconnection runs)
      run m1 "H_$pol" "$prec" "$ranks" "$dec" $common eps0=0.5 gamma_max=6.5 tau=0.0906 gamma_p=6.0 steps=40 policy=$pol
      run m1 "L_$pol" "$prec" "$ranks" "$dec" $common eps0=1.8 gamma_max=0.0 tau=0.3378 gamma_p=1.2 steps=100 policy=$pol
    done
  done
}

m2_cases() { # precision
  local prec="$1" cfg ranks dec hot
  for cfg in "2:2, 1" "4:2, 2"; do
    ranks="${cfg%%:*}"; dec="${cfg#*:}"
    # V1, unchanged (scripts/run_v1.sh), with the lines per rank added
    run m2 A_on        "$prec" "$ranks" "$dec" order=leptons_first policy=on  rank_lines=true
    run m2 A_off       "$prec" "$ranks" "$dec" order=leptons_first policy=off rank_lines=true
    run m2 B_on        "$prec" "$ranks" "$dec" order=ions_first    policy=on  rank_lines=true
    run m2 B_off       "$prec" "$ranks" "$dec" order=ions_first    policy=off rank_lines=true
    run m2 A_on_stale  "$prec" "$ranks" "$dec" order=leptons_first policy=on  stale_prev=true rank_lines=true
    run m2 B_on_stale  "$prec" "$ranks" "$dec" order=ions_first    policy=on  stale_prev=true rank_lines=true
    # the write left out, stale slots: leptons first (the negative control) and ions first
    run m2 A_neg_stale "$prec" "$ranks" "$dec" order=leptons_first policy=negative stale_prev=true rank_lines=true
    run m2 B_neg_stale "$prec" "$ranks" "$dec" order=ions_first    policy=negative stale_prev=true rank_lines=true
    # every emitter on rank 0: the leptons found on the other ranks after the
    # first step crossed a rank boundary in the step of their birth
    hot="0.0, 0.5, 0.0, 1.0"; [ "$ranks" = 4 ] && hot="0.0, 0.5, 0.0, 0.5"
    run m2 B_birth     "$prec" "$ranks" "$dec" order=ions_first    policy=on  rank_lines=true hot_fraction=1.0 "hot_box=$hot" dump_step=0
    run m2 B_birth_off "$prec" "$ranks" "$dec" order=ions_first    policy=off rank_lines=true hot_fraction=1.0 "hot_box=$hot" dump_step=0
    run m2 A_birth     "$prec" "$ranks" "$dec" order=leptons_first policy=on  rank_lines=true hot_fraction=1.0 "hot_box=$hot" dump_step=0
    run m2 A_birth_off "$prec" "$ranks" "$dec" order=leptons_first policy=off rank_lines=true hot_fraction=1.0 "hot_box=$hot" dump_step=0
  done
}

m3_cases() { # precision [suffix of the group, for the bounds-checked build]
  local prec="$1" grp="m3${2:-}"
  local common="order=ions_first fields=false neutralizer=false gauss=false isotropic=true n_ions=200000 gamma_p=6.0 eps0=0.5 gamma_max=6.5 J_ref=0.389958 p_ref=0.05 steps=10 diag_interval=1 rank_lines=true maxnpart_leptons=30000 sort_interval=0 hot_fraction=0.9"
  # nine tenths of the emitters well inside rank 0, the others well inside the last rank;
  # nothing reaches a rank boundary in 10 steps (3.6 cells)
  run "$grp" full "$prec" 2 "2, 1" $common "hot_box=0.1, 0.4, 0.1, 0.9" "cold_box=0.6, 0.9, 0.1, 0.9"
  run "$grp" full "$prec" 4 "2, 2" $common "hot_box=0.1, 0.4, 0.1, 0.4" "cold_box=0.6, 0.9, 0.6, 0.9"
}
