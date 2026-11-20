#!/usr/bin/env python3
"""Mutated copies of two headers of the pull-request branch, as include overlays, and a diagnostic
copy of the unit test. Nothing in the worktree is changed.

    make_mutants.py <worktree of dev/bh-emission> <output directory>

overlays/<name>/archetypes/emission/bethe_heitler.h   the policy with part of the previous-position
                                                      write removed (seven mutations), or with the
                                                      finite check of the timestep removed
overlays/no_mass_guard/kernels/pushers/sr_policies.h  the factory without the finite check of the mass
diagnostic/bethe_heitler_diagnostic.cpp               the unit test with `check` counting and printing
                                                      a failure where the shipped test throws, and
                                                      every subtest run; assertions and limits are
                                                      those of the shipped test
"""
import os
import subprocess
import sys

wt, out = sys.argv[1:3]
assert subprocess.run(["git", "-C", wt, "status", "--porcelain"], capture_output=True, text=True).stdout == ""
head = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
policy = open(os.path.join(wt, "src/archetypes/emission/bethe_heitler.h")).read()
factory = open(os.path.join(wt, "src/kernels/pushers/sr_policies.h")).read()
test = open(os.path.join(wt, "tests/archetypes/bethe_heitler.cpp")).read()

block = """      if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i1_prev(p)  = xi_Cd[0];
        dx1_prev(p) = dxi_Cd[0];
      }
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i2_prev(p)  = xi_Cd[1];
        dx2_prev(p) = dxi_Cd[1];
      }
      if constexpr (M::Dim == Dim::_3D) {
        i3_prev(p)  = xi_Cd[2];
        dx3_prev(p) = dxi_Cd[2];
      }
"""
assert policy.count(block) == 1


def without(lines_to_drop):
    kept = [l for l in block.splitlines(keepends=True) if not any(k in l for k in lines_to_drop)]
    return "".join(kept)


def only_for(species):
    body = "".join("  " + l for l in block.splitlines(keepends=True))
    return "      if (i1_prev.data() == %s_i1_prev.data()) {\n%s      }\n" % (species, body)


integer = ["i1_prev(p)", "i2_prev(p)", "i3_prev(p)"]
fractional = ["dx1_prev(p)", "dx2_prev(p)", "dx3_prev(p)"]
mutations = {
    "no_write": without(integer + fractional),
    "integer_only": without(fractional),
    "fractional_only": without(integer),
    "electrons_only": only_for("electron"),
    "positrons_only": only_for("positron"),
    "x_only": without(["i2_prev(p)", "dx2_prev(p)", "i3_prev(p)", "dx3_prev(p)"]),
    "y_only": without(["i1_prev(p)", "dx1_prev(p)", "i3_prev(p)", "dx3_prev(p)"]),
}
for name, new in mutations.items():
    assert new != block
    d = os.path.join(out, "overlays", name, "archetypes/emission")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "bethe_heitler.h"), "w").write(policy.replace(block, new))

guard = """      raise::ErrorIf(not std::isfinite(dt),
                     "Bethe-Heitler: the timestep must be finite",
                     HERE);
"""
assert policy.count(guard) == 1
d = os.path.join(out, "overlays", "no_dt_guard", "archetypes/emission")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "bethe_heitler.h"), "w").write(policy.replace(guard, ""))

guard = """      raise::ErrorIf(not(pusher_ctx.mass > 0.0f) or
                       not std::isfinite(pusher_ctx.mass),
"""
assert factory.count(guard) == 1
d = os.path.join(out, "overlays", "no_mass_guard", "kernels/pushers")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "sr_policies.h"), "w").write(
    factory.replace(guard, "      raise::ErrorIf(not(pusher_ctx.mass > 0.0f),\n"))

check = """  void check(bool condition, const std::string& msg) {
    if (not condition) {
      throw std::runtime_error(msg);
    }
  }
"""
assert test.count(check) == 1
diag = test.replace(check, """  int n_failed_checks = 0;

  // diagnostic copy: a failed check is printed and counted, and the test goes on
  void check(bool condition, const std::string& msg) {
    if (not condition) {
      ++n_failed_checks;
      std::cout << "FAILED CHECK: " << msg << std::endl;
    }
  }
""")
main = """  try {
    test_charge();
    test_charge_reused_slots();
    test_rate_and_loss();
    test_refusals_and_capacity();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
"""
assert diag.count(main) == 1
diag = diag.replace(main, """  for (const auto subtest :
       { &test_charge, &test_charge_reused_slots, &test_rate_and_loss, &test_refusals_and_capacity }) {
    try {
      subtest();
    } catch (std::exception& e) {
      ++n_failed_checks;
      std::cout << "EXCEPTION: " << e.what() << std::endl;
    }
  }
  std::cout << "failed checks: " << n_failed_checks << std::endl;
  GlobalFinalize();
  return n_failed_checks == 0 ? 0 : 1;
""")
os.makedirs(os.path.join(out, "diagnostic"), exist_ok=True)
open(os.path.join(out, "diagnostic", "bethe_heitler_diagnostic.cpp"), "w").write(diag)
open(os.path.join(out, "source_head.txt"), "w").write(head + "\n")
print("head", head, "mutations:", " ".join(list(mutations) + ["no_dt_guard", "no_mass_guard"]))
