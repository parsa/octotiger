#include "defs.hpp"
#include "geometry.hpp"
#include "options.hpp"

#include "../common_kernel/helper.hpp"
#include "calculate_stencil.hpp"

#include <fstream>

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        two_phase_stencil calculate_stencil(void) {
            std::array<two_phase_stencil, 8> stencils;

            // used to check the radiuses of the outer and inner sphere
            const real theta0 = opts.theta;

            // int64_t i0 = 0;
            // int64_t i1 = 0;
            // int64_t i2 = 0;

            for (int64_t i0 = 0; i0 < 2; ++i0) {
                for (int64_t i1 = 0; i1 < 2; ++i1) {
                    for (int64_t i2 = 0; i2 < 2; ++i2) {
                        two_phase_stencil stencil;
                        for (int64_t j0 = i0 - INX; j0 < i0 + INX; ++j0) {
                            for (int64_t j1 = i1 - INX; j1 < i1 + INX; ++j1) {
                                for (int64_t j2 = i2 - INX; j2 < i2 + INX; ++j2) {
                                    // don't interact with yourself!
                                    if (i0 == j0 && i1 == j1 && i2 == j2) {
                                        continue;
                                    }

                                    // indices on coarser level (for outer stencil boundary)
                                    const int64_t i0_c = (i0 + INX) / 2 - INX / 2;
                                    const int64_t i1_c = (i1 + INX) / 2 - INX / 2;
                                    const int64_t i2_c = (i2 + INX) / 2 - INX / 2;

                                    const int64_t j0_c = (j0 + INX) / 2 - INX / 2;
                                    const int64_t j1_c = (j1 + INX) / 2 - INX / 2;
                                    const int64_t j2_c = (j2 + INX) / 2 - INX / 2;

                                    const real theta_f =
                                        detail::reciprocal_distance(i0, i1, i2, j0, j1, j2);
                                    const real theta_c = detail::reciprocal_distance(
                                        i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                                    // not in inner sphere (theta_c > theta0), but in outer
                                    // sphere
                                    if (theta_c > theta0 && theta_f <= theta0) {
                                        stencil.stencil_elements.emplace_back(
                                            j0 - i0, j1 - i1, j2 - i2);
                                        stencil.stencil_phase_indicator.emplace_back(true);
                                    } else if (theta_c > theta0) {
                                        stencil.stencil_elements.emplace_back(
                                            j0 - i0, j1 - i1, j2 - i2);
                                        stencil.stencil_phase_indicator.emplace_back(false);
                                    }
                                }
                            }
                        }
                        stencils[i0 * 4 + i1 * 2 + i2] = stencil;
                    }
                }
            }

            two_phase_stencil superimposed_stencil;
            for (size_t i = 0; i < 8; i++) {
                for (auto element_index = 0; element_index < stencils[i].stencil_elements.size();
                     ++element_index) {
                    multiindex<>& stencil_element = stencils[i].stencil_elements[element_index];
                    bool found = false;
                    for (multiindex<>& super_element : superimposed_stencil.stencil_elements) {
                        if (stencil_element.compare(super_element)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        superimposed_stencil.stencil_elements.push_back(stencil_element);
                        superimposed_stencil.stencil_phase_indicator.push_back(
                            stencils[i].stencil_phase_indicator[element_index]);
                    }
                }
                std::cout << "Stencil size: " << stencils[i].stencil_elements.size() << std::endl;
            }

            std::ofstream out("stencil.tex");
            // print latex header
            out << "\\documentclass[tikz,convert={outfile=\\jobname.svg}]{standalone}" << std::endl
                << "\\usepackage{tikz,tikz-3dplot}" << std::endl
                << "\\usetikzlibrary{arrows,decorations.pathmorphing,backgrounds,positioning,fit,"
                   "petri}"
                << std::endl
                << "\\usetikzlibrary{arrows, decorations.markings}" << std::endl
                << "\\usetikzlibrary{shapes.geometric}" << std::endl
                << "\\usetikzlibrary{shapes,calc}" << std::endl
                << "\\usetikzlibrary{decorations.pathreplacing}" << std::endl
                << std::endl;
            out << "\\newcommand{\\squarex}[5]{" << std::endl
                << "\\filldraw[fill=#4,draw=#5,opacity=1.0] (0+#1,0+#2,#3) -- (1+#1,0+#2,#3) -- "
                   "(1+#1,1+#2,#3) -- (0+#1,1+#2,#3) -- (0+#1,0+#2,#3);"
                << std::endl
                << "}" << std::endl;
            out << "\\newcommand{\\squarey}[5]{" << std::endl
                << "\\filldraw[fill=#4,draw=#5,opacity=1.0] (#1,#2,#3) -- (#1,1+#2,#3) -- "
                   "(#1,1+#2,1+#3) -- (#1,#2,1+#3) -- (#1,#2,#3);"
                << std::endl
                << "}" << std::endl;

            out << "\\newcommand{\\cube}[7]{" << std::endl
                << "\\coordinate (O) at (#1,#2,#3);" << std::endl
                << "\\coordinate (A) at (#1,#2+#4,#3);" << std::endl
                << "\\coordinate (B) at (#1,#2+#4,#3+#4);" << std::endl
                << "\\coordinate (C) at (#1,#2,#3+#4);" << std::endl
                << "\\coordinate (D) at (#1+#4,#2,#3);" << std::endl
                << "\\coordinate (E) at (#1+#4,#2+#4,#3);" << std::endl
                << "\\coordinate (F) at (#1+#4,#2+#4,#3+#4);" << std::endl
                << "\\coordinate (G) at (#1+#4,#2,#3+#4);" << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (O) -- (C) -- (G) -- (D) -- cycle;"
                << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (O) -- (A) -- (E) -- (D) -- cycle;"
                << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (O) -- (A) -- (B) -- (C) -- cycle;"
                << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (D) -- (E) -- (F) -- (G) -- cycle;"
                << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (C) -- (B) -- (F) -- (G) -- cycle;"
                << std::endl
                << "\\draw[blue,fill=#5,opacity=#6,draw=#7] (A) -- (B) -- (F) -- (E) -- cycle;"
                << std::endl
                << "}" << std::endl;

            out << "\\begin{document}" << std::endl;
            auto draw_slice = [&out, &superimposed_stencil](int centerz) {
                out << "\\begin{tikzpicture}" << std::endl;
                int z = 0;
                int centerx = 6;
                int centery = 7;
                for (int y = 0; y < 16; ++y) {
                    for (int x = 0; x < 16; ++x) {
                        int xdist = x - centerx;
                        int ydist = y - centery;
                        if (xdist == 0 && ydist == 0 && centerz == 0) {
                            out << "\\squarex{" << x << "-1}{" << y << "-1}{" << z
                                << " }{brown!80}{black}" << std::endl;
                            continue;
                        }
                        multiindex<> current_element(xdist, ydist, centerz);
                        bool found = false;
                        for (multiindex<>& super_element : superimposed_stencil.stencil_elements) {
                            if (current_element.compare(super_element)) {
                                found = true;
                                break;
                            }
                        }
                        if (found) {
                            out << "\\squarex{" << x << "-1}{" << y << "-1}{" << z
                                << " }{brown!35}{black}" << std::endl;
                        } else {
                            if (x > 3 && x < 12 && y > 3 && y < 12) {
                                out << "\\squarex{" << x << "-1}{" << y << "-1}{" << z
                                    << " }{white}{gray}" << std::endl;
                            } else {
                                out << "\\squarex{" << x << "-1}{" << y << "-1}{" << z
                                    << " }{gray!30}{gray}" << std::endl;
                            }
                        }
                    }
                }
                out << "\\draw[line width=2,draw=black!75] (3,3,-0) -- (3+8,3,-0) -- (3+8,3+8,-0) -- "
                       "(3,3+8,-0) -- (3,3,-0);"
                    << std::endl;
                out << "\\end{tikzpicture}" << std::endl;
            };
            auto draw_whole_stencil = [&out, &superimposed_stencil](void) {
                out << "\\begin{tikzpicture}" << std::endl;
                int z = 0;
                int centerx = 4;
                int centery = 4;
                int centerz = 4;
                for (int z = 0; z < 10; ++z) {
                    for (int y = 0; y < 10; ++y) {
                        for (int x = 0; x < 10; ++x) {
                            int xdist = x - centerx;
                            int ydist = y - centery;
                            int zdist = z - centerz;
                            multiindex<> current_element(xdist, ydist, zdist);
                            bool found = false;
                            for (multiindex<>& super_element :
                                superimposed_stencil.stencil_elements) {
                                if (current_element.compare(super_element)) {
                                    found = true;
                                    break;
                                }
                            }
                            if (found) {
                                // out << "\\squarex{" << x << "-1}{" << y << "-1}{" << z
                                //     << " }{green!30}{gray}" << std::endl;
                                float transparency = 1.0;
                                if (transparency < 0.0)
                                    transparency = 0.0;
                                if (zdist == 0)
                                    out << "\\cube{" << x << "}{" << y << "}{" << z
                                        << "}{1}{brown!20}{1.0}{black}" << std::endl;
                                else
                                    out << "\\cube{" << x << "}{" << y << "}{" << z
                                        << "}{1}{brown!20}{" << transparency << "}{black}"
                                        << std::endl;
                            } else {
                                //     float transparency = 1.0 - 1.0 / 9.0*z;
                                // if (transparency < 0.0)
                                //     transparency = 0.0;
                                // out << "\\cube{" << x << "}{" << y << "}{" << z <<
                                // "}{1}{gray!30}{0.2}{black!60}" << std::endl;
                            }
                        }
                    }
                }
                out << "\\end{tikzpicture}" << std::endl;
            };

            // for (int i = -6; i < 8; ++i) {
            //     draw_slice(i);
            // }
            draw_whole_stencil();
            //draw_slice(0);

            out << "\\end{document}" << std::endl;
            out.close();

            return superimposed_stencil;
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
