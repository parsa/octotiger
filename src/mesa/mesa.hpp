#include <string>
#include <functional>

struct mesa_eos_t {
	std::function<double(double)> p_of_rho;
	std::function<double(double)> h_of_rho;
	std::function<double(double)> rho_of_h;
};

mesa_eos_t build_eos_from_mesa(const std::string& filename);
