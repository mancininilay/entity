    void write_pairs(Domain<S, M>& domain) {
      if (pair_minus == 0) {
        return;
      }
      auto& spm = domain.species[pair_minus - 1];
      auto& spp = domain.species[pair_plus - 1];
      std::ofstream out(per_rank_name("hadronic_pairs"), std::ios::trunc);
      out << "# bh_proto pairs: species e- = " << (int)pair_minus << ", e+ = " << (int)pair_plus
          << "; row k holds slot k of e- (k < npart_minus) and slot k of e+ (k < npart_plus), each "
             "species complete with its own count; a side beyond its count is written with tag -1 "
             "and zeros. The two sides of a row are partners only while neither species has "
             "migrated or been sorted: sum each species on its own. Background particles have "
             "payload birth time 0\n";
      out << "# columns: k | i1- | dx1- | i1+ | dx1+ | w- | w+ | ux1- | ux2- | ux3- | ux1+ | ux2+ | ux3+ | "
             "tag- | tag+ | birth- | gamma_p- | eps- | birth+ | gamma_p+ | eps+ | i1_prev- | dx1_prev- "
             "| i1_prev+ | dx1_prev+\n";
      const npart_t n_m = spm.npart(), n_p = spp.npart(), n = std::max(n_m, n_p);
      auto mirror = [](const auto& v) {
        auto h = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(h, v);
        return h;
      };
      auto i1m = mirror(spm.i1), i1p = mirror(spp.i1);
      auto dxm = mirror(spm.dx1), dxp = mirror(spp.dx1);
      auto i1pm = mirror(spm.i1_prev), i1pp = mirror(spp.i1_prev);
      auto dxpm = mirror(spm.dx1_prev), dxpp = mirror(spp.dx1_prev);
      auto wm = mirror(spm.weight), wp = mirror(spp.weight);
      auto u1m = mirror(spm.ux1), u2m = mirror(spm.ux2), u3m = mirror(spm.ux3);
      auto u1p = mirror(spp.ux1), u2p = mirror(spp.ux2), u3p = mirror(spp.ux3);
      auto tm = mirror(spm.tag), tp = mirror(spp.tag);
      auto pm = mirror(spm.pld_r), pp = mirror(spp.pld_r);
      out << "# n " << n << " npart_minus " << n_m << " npart_plus " << n_p << "\n";
      out << std::setprecision(17);
      for (npart_t k = 0; k < n; ++k) {
        // one side's value of slot k; beyond that species' own count the slot does not exist
        const bool hm = k < n_m, hp = k < n_p;
        auto m_side = [hm](auto v) { return hm ? v : decltype(v) {}; };
        auto p_side = [hp](auto v) { return hp ? v : decltype(v) {}; };
        out << k << " " << m_side(i1m(k)) << " " << m_side((double)dxm(k)) << " " << p_side(i1p(k)) << " "
            << p_side((double)dxp(k)) << " " << m_side((double)wm(k)) << " " << p_side((double)wp(k)) << " "
            << m_side((double)u1m(k)) << " " << m_side((double)u2m(k)) << " " << m_side((double)u3m(k)) << " "
            << p_side((double)u1p(k)) << " " << p_side((double)u2p(k)) << " " << p_side((double)u3p(k)) << " "
            << (hm ? tm(k) : static_cast<short>(-1)) << " " << (hp ? tp(k) : static_cast<short>(-1)) << " "
            << m_side((double)pm(k, 0)) << " " << m_side((double)pm(k, 1)) << " " << m_side((double)pm(k, 2)) << " "
            << p_side((double)pp(k, 0)) << " " << p_side((double)pp(k, 1)) << " " << p_side((double)pp(k, 2)) << " "
            << m_side(i1pm(k)) << " " << m_side((double)dxpm(k)) << " " << p_side(i1pp(k)) << " "
            << p_side((double)dxpp(k)) << "\n";
      }
    }
