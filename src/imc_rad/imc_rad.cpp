/*
 * imc_rad.cpp
 *
 *  Created on: Jan 30, 2018
 *      Author: dmarce1
 */

#include "imc_rad.hpp"
#include "physcon.hpp"
#include "space_vector.hpp"
#include "util.hpp"
#include <functional>
#include <stdlib.h>

const real& h = physcon.h;
const real& c = physcon.c;
const real& kB = physcon.kb;

inline real cosh(real x) {
	return real(0.5) * (exp(x) + exp(-x));
}

real planck(real nu, real T) {
	return (real(2) * h * std::pow(nu, 3)) / ((c * c) * (exp((h * nu) / (kB * T)) - real(1)));
}

real dplanck_dT(real nu, real T) {
	return (h * h * std::pow(nu, 4)) / ((kB * c * c * T * T) * (cosh((h * nu) / (kB * T)) - real(1)));
}

real integral(const std::function<real(real)>& f, double a, double b, int N) {
	real sum;
	real x;
	real dx = (b - a) / real(N);
	sum = (f(a) + f(b)) * dx;
	for (int i = 1; i < N - 1; i++) {
		x = a + double(i) * dx;
		sum += f(x) * dx;
	}
	return sum;
}

real planck_mean_opacity(const std::function<real(real, real)>& kappa, real nu1, real nu2, real T) {
	const integer N = 100;
	const std::function<real(real)> fn = [T,kappa](real nu) {
		return planck(nu,T) * kappa(nu,T);
	};
	const std::function<real(real)> fd = [T](real nu) {
		return planck(nu,T);
	};
	return integral(fn, nu1, nu2, N) / integral(fd, nu1, nu2, N);

}

real rosseland_mean_opacity(const std::function<real(real, real)>& kappa, real nu1, real nu2, real T) {
	const integer N = 100;
	const std::function<real(real)> fn = [T](real nu) {
		return dplanck_dT(nu,T);
	};
	const std::function<real(real)> fd = [T,kappa](real nu) {
		return dplanck_dT(nu,T) / kappa(nu,T);
	};
	return integral(fn, nu1, nu2, N) / integral(fd, nu1, nu2, N);

}

struct particle {
private:
	static void beta2_gamma(const space_vector& beta, real& beta2, real& gamma) {
		beta2 = beta.abs2();
		if (beta2 >= 1.0) {
			printf("Superluminal, beta = %e\n", std::sqrt(beta2));
			abort();
		}
		gamma = 1.0 / std::sqrt(1.0 - beta2);
	}
	void frame_transform(const space_vector& v) {
		real beta2, gamma;
		space_vector beta = v / c;
		beta2_gamma(beta, beta2, gamma);
		const real nu = p.abs();
		p += beta * beta.dot(p) * ((gamma * gamma) / (gamma + 1.0));
		p += beta * gamma * nu;
	}
public:
	space_vector x;
	space_vector p;
	real n;

	real frequency() const {
		return p.abs();
	}

	space_vector direction() const {
		return p / p.abs();
	}

	real distance_to_collision(const space_vector& v, real sigma) const {
		real beta2, gamma, r, dx;
		space_vector beta = v / c;
		beta2_gamma(beta, beta2, gamma);
		dx = gamma * (1.0 + p.dot(beta) / p.abs()) / sigma;
		r = rand1();
		return -log(r) * dx;
	}

	void propagate(real dt) {
		x += (p / p.abs()) * c * dt;
	}

};

