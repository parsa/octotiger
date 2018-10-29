/*
 * silo.cpp
 *
 *  Created on: Oct 16, 2018
 *      Author: dmarce1
 */

#include "node_registry.hpp"
#include "silo.hpp"
#include "options.hpp"
#include "physcon.hpp"
#include <hpx/lcos/broadcast.hpp>
#include <future>
#include "util.hpp"

static const auto& localities = options::all_localities;

static std::mutex silo_mtx_;

void output_stage1(std::string fname, int cycle);
std::vector<node_location::node_id> output_stage2(std::string fname, int cycle);
void output_stage3(std::string fname, int cycle);

HPX_PLAIN_ACTION(output_stage1, output_stage1_action);
HPX_PLAIN_ACTION(output_stage2, output_stage2_action);
HPX_PLAIN_ACTION(output_stage3, output_stage3_action);

#include <hpx/include/threads.hpp>
#include <hpx/include/run_as.hpp>

template<class T>
struct db_type {
	static constexpr int d = DB_UNKNOWN;
};

template<>
struct db_type<integer> {
	static constexpr int d = DB_LONG_LONG;
};

template<>
struct db_type<real> {
	static constexpr int d = DB_DOUBLE;
};

template<>
struct db_type<char> {
	static constexpr int d = DB_CHAR;
};

constexpr int db_type<integer>::d;
constexpr int db_type<char>::d;
constexpr int db_type<real>::d;

template<class T>
struct write_silo_var {
	void operator()(DBfile* db, const char* name, T var) {
		int one = 1;
		DBWrite(db, name, &var, &one, 1, db_type<T>::d);
	}
};

template<class T>
struct read_silo_var {
	T operator()(DBfile* db, const char* name) {
		int one = 1;
		T var;
		if (DBReadVar(db, name, &var) != 0) {
			std::cout << "Unable to read " << name << " \n";
		}
		return var;
	}
};

struct mesh_vars_t {
	std::vector<silo_var_t> vars;
	std::vector<std::string> var_names;
	std::vector<std::pair<std::string, real>> outflow;
	std::string mesh_name;
	std::vector<std::vector<real>> X;
	std::array<int, NDIM> X_dims;
	std::array<int, NDIM> var_dims;
	node_location location;
	mesh_vars_t(mesh_vars_t&&) = default;
	int compression_level() {
		int lev = 0;
		int i = var_dims[0];
		while (i > INX) {
			i >>= 1;
		}
		return lev;
	}
	mesh_vars_t(const node_location& loc, int compression = 0) :
			location(loc), X(NDIM){
		const int nx = (INX << compression);
		X_dims[0] = X_dims[1] = X_dims[2] = nx + 1;
		var_dims[0] = var_dims[1] = var_dims[2] = nx;
		const real dx = 2.0 * opts.xscale / INX / (1 << loc.level());
		for (int d = 0; d < NDIM; d++) {
			X[d].resize(X_dims[d]);
			const real o = loc.x_location(d);
			for (int i = 0; i < X_dims[d]; i++) {
				X[d][i] = o + i * dx;
			}
		}
		mesh_name = std::to_string(loc.to_id());
	}
};

static std::vector<hpx::future<mesh_vars_t>> futs_;
static std::vector<node_location::node_id> loc_ids;

void output_stage1(std::string fname, int cycle) {

	std::vector<node_location::node_id> ids;
	futs_.clear();
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		if (!i->second->refined()) {
			futs_.push_back(hpx::async([](node_location loc, node_server* this_ptr)
			{
				const real dx = TWO / real(1 << loc.level()) / real(INX);
				mesh_vars_t rc(loc);
				const std::string suffix = std::to_string(loc.to_id());
				rc.vars = std::move(this_ptr->get_hydro_grid().var_data(suffix));
				rc.outflow = std::move(this_ptr->get_hydro_grid().get_outflows());
				return std::move(rc);
			}, i->first, i->second));
		}
	}
}

static const int HOST_NAME_LEN = 100;
static int epoch = 0;
static time_t start_time = time(NULL);
static integer start_step = 0;
static int timestamp;
static int nsteps;
static int time_elapsed;
static int steps_elapsed;

std::vector<mesh_vars_t> compress(std::vector<mesh_vars_t>&& mesh_vars) {
	using table_type = std::unordered_map<node_location::node_id, std::shared_ptr<mesh_vars_t>>;
	table_type table;
	std::vector<mesh_vars_t> done;
	for (auto& mv : mesh_vars) {
		table.insert(std::make_pair(mv.location.to_id(), std::make_shared<mesh_vars_t>(std::move(mv))));
	}
	int ll = 0;
	std::vector<table_type::iterator> iters;
	while (table.size()) {
		auto i = table.begin();
		auto loc_bits = i->first & node_location::node_id(~0x7);
		for (int j = 0; j < NCHILD; j++) {
			auto this_iter = table.find(loc_bits | j);
			if (this_iter != table.end()) {
				iters.push_back(this_iter);
			}
		}
		const auto field_names = grid::get_field_names();
		const int nfields = field_names.size();
		if (iters.size() == NCHILD) {
			printf( "Compress\n");
			node_location ploc;
			ploc.from_id(loc_bits);
			ploc = ploc.get_parent();
			auto new_mesh_ptr = std::make_shared<mesh_vars_t>(ploc, iters[0]->second->compression_level() + 1);
			for (int f = 0; f < nfields; f++) {
				silo_var_t new_var(field_names[f] + "_" + std::to_string(ploc.to_id()), new_mesh_ptr->var_dims[0]);
				for (integer ci = 0; ci != NCHILD; ci++) {
					const int nx = new_mesh_ptr->var_dims[0] / 2;
					const int ib = ((ci >> 0) & 1) * nx;
					const int jb = ((ci >> 1) & 1) * nx;
					const int kb = ((ci >> 2) & 1) * nx;
					const int ie = ib + nx;
					const int je = jb + nx;
					const int ke = kb + nx;
					for (int i = ib; i < ie; i++) {
						for (int j = jb; j < je; j++) {
							for (int k = kb; k < ke; k++) {
								const int iiip = i * (4 * nx * nx) + j * 2 * nx + k;
								const int iiic = (i - ib) * (nx * nx) + (j - jb) * nx + (k - kb);
								new_var(iiip) = iters[ci]->second->vars[f](iiic);
							}
						}
					}
				}
				new_mesh_ptr->vars.push_back(std::move(new_var));
			}
			for (auto this_iter : iters) {
				table.erase(this_iter);
			}
			table.insert(std::make_pair(ploc.to_id(), new_mesh_ptr));
		} else {
			printf( "------------------\n", iters.size());
			for (auto this_iter : iters) {
				node_location loc;
				loc.from_id(this_iter->first);
				printf("%s %o\n", loc.to_str().c_str(), this_iter->first);
				done.push_back(std::move(*this_iter->second));
				table.erase(this_iter);
			}
		}
		iters.clear();
	}
	return std::move(done);
}

static std::vector<mesh_vars_t> all_mesh_vars;

std::vector<node_location::node_id>  output_stage2(std::string fname, int cycle) {
	const int this_id = hpx::get_locality_id();
	const int nfields = grid::get_field_names().size();
	std::string this_fname = fname + std::string(".silo");
	double dtime = node_registry::begin()->second->get_time();
	all_mesh_vars.clear();
	for (auto& this_fut : futs_) {
		all_mesh_vars.push_back(std::move(GET(this_fut)));
	}
	std::vector<node_location::node_id> ids;
	all_mesh_vars = compress(std::move(all_mesh_vars));
	for (const auto& mv : all_mesh_vars) {
		ids.push_back(mv.location.to_id());
	}

	return ids;
}

void output_stage3(std::string fname, int cycle) {
	const int this_id = hpx::get_locality_id();
	const int nfields = grid::get_field_names().size();
	std::string this_fname = fname + std::string(".silo");
	double dtime = node_registry::begin()->second->get_time();
	GET( hpx::threads::run_as_os_thread( [&this_fname,this_id,&all_mesh_vars,&dtime](integer cycle) {
		DBfile *db;
		if (this_id == 0) {
			db = DBCreateReal(this_fname.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-tiger", DB_PDB);
		} else {
			db = DBOpenReal(this_fname.c_str(), DB_PDB, DB_APPEND);
		}
		float ftime = dtime;
		auto optlist = DBMakeOptlist(5);
		int one = 1;
		int opt1 = DB_CARTESIAN;
		DBAddOption(optlist, DBOPT_HIDE_FROM_GUI, &one);
		DBAddOption(optlist, DBOPT_COORDSYS, &opt1);
		DBAddOption(optlist, DBOPT_CYCLE, &cycle);
		DBAddOption(optlist, DBOPT_DTIME, &dtime);
		DBAddOption(optlist, DBOPT_TIME, &ftime);
		const char* coord_names[] = {"x", "y", "z"};
		constexpr int data_type = DB_DOUBLE;
		constexpr int ndim = NDIM;
		constexpr int coord_type = DB_COLLINEAR;
		for( const auto& mesh_vars : all_mesh_vars)
		{
			const auto& X = mesh_vars.X;
			const real* coords[] = {X[0].data(), X[1].data(), X[2].data()};
			DBPutQuadmesh(db, mesh_vars.mesh_name.c_str(), coord_names, coords, mesh_vars.X_dims.data(), ndim, data_type, coord_type, optlist);
			for (const auto& o : mesh_vars.vars) {
				DBPutQuadvar1(db, o.name(), mesh_vars.mesh_name.c_str(), o.data(), mesh_vars.var_dims.data(), ndim, (const void*) NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist);
			}
			for (const auto& p : mesh_vars.outflow) {
				const std::string name = p.first + std::string("_outflow_") + mesh_vars.mesh_name; write_silo_var<real>()(db, name.c_str(), p.second);
			}
		}
		DBFreeOptlist( optlist); DBClose( db);
	}, cycle));
	if (this_id < integer(localities.size()) - 1) {
		output_stage3_action func;
		func(localities[this_id + 1], fname, cycle);
	}

	double rtime = node_registry::begin()->second->get_rotation_count();
	if (this_id == 0) {
		GET( hpx::threads::run_as_os_thread([&this_fname,nfields,&rtime](int cycle) {
				auto* db = DBOpenReal(this_fname.c_str(), DB_PDB, DB_APPEND);
				double dtime = node_registry::begin()->second->get_time();
				float ftime = dtime;
				std::vector<node_location> node_locs;
				std::vector<char*> mesh_names;
				std::vector<std::vector<char*>> field_names(nfields);
				for (auto& i : loc_ids) {
					node_location nloc;
					nloc.from_id(i);
					node_locs.push_back(nloc);

				}
				const auto top_field_names = grid::get_field_names();
				for (int i = 0; i < node_locs.size(); i++) {
					const auto suffix = std::to_string(node_locs[i].to_id());
					const auto str = suffix;
					char* ptr = new char[str.size() + 1];
					std::strcpy(ptr, str.c_str());
					mesh_names.push_back(ptr);
					for (int f = 0; f < nfields; f++)
					{
						const auto str = top_field_names[f] + std::string("_") + suffix;
						char* ptr = new char[str.size() + 1];
						strcpy(ptr, str.c_str());
						field_names[f].push_back(ptr);
					}
				}

				const int n_total_domains = mesh_names.size();
				static const std::vector<int> meshtypes(n_total_domains, DB_QUAD_RECT);
				static const std::vector<int> datatypes(n_total_domains, DB_QUADVAR);

				auto optlist = DBMakeOptlist(4);
				int opt1 = DB_CARTESIAN;
				DBAddOption(optlist, DBOPT_COORDSYS, &opt1);
				DBAddOption(optlist, DBOPT_CYCLE, &cycle);
				DBAddOption(optlist, DBOPT_DTIME, &dtime);
				DBAddOption(optlist, DBOPT_TIME, &ftime);
				assert( n_total_domains > 0 );
				DBPutMultimesh(db, "mesh", n_total_domains, mesh_names.data(), meshtypes.data(), optlist);
				for (int f = 0; f < nfields; f++) {
					DBPutMultivar( db, top_field_names[f].c_str(), n_total_domains, field_names[f].data(), datatypes.data(), optlist);
				}
				DBFreeOptlist( optlist);
				for (auto ptr : mesh_names) {
					delete[] ptr;
				}
				for (int f = 0; f < nfields; f++) {
					for (auto& s : field_names[f]) {
						delete[] s;
					}
				}
				write_silo_var<integer> fi; write_silo_var<real> fr;

				fi(db, "n_species", integer(opts.n_species));
				fi(db, "eos", integer(opts.eos));
				fi(db, "gravity", integer(opts.gravity));
				fi(db, "hydro", integer(opts.hydro));
				fr(db, "omega", grid::get_omega());
				fr(db, "output_frequency", opts.output_dt);
				fi(db, "problem", integer(opts.problem));
				fi(db, "radiation", integer(opts.radiation));
				fr(db, "refinement_floor", opts.refinement_floor);
				fr(db, "time", dtime);
				fr(db, "rotational_time", rtime);
				fr(db, "xscale", opts.xscale);
				char hostname[HOST_NAME_LEN];
				real g, cm, s, K;
				these_units(g,cm,s,K);
				fr(db, "g", g);
				fr(db, "cm", cm);
				fr(db, "s", s);
				fr(db, "K", K);
				gethostname(hostname, HOST_NAME_LEN);
				DBWrite( db, "hostname", hostname, &HOST_NAME_LEN, 1, DB_CHAR);
				write_silo_var<integer>()(db, "timestamp", timestamp);
				write_silo_var<integer>()(db, "epoch", epoch);
				write_silo_var<integer>()(db, "N_localities", localities.size());
				write_silo_var<integer>()(db, "step_count", nsteps);
				write_silo_var<integer>()(db, "time_elapsed", time_elapsed);
				write_silo_var<integer>()(db, "steps_elapsed", steps_elapsed);
				DBClose( db);
			}, cycle));
	}
}

void output_all(std::string fname, int cycle, bool block) {
	block = true;
//	static hpx::lcos::local::spinlock mtx;
//	std::lock_guard<hpx::lcos::local::spinlock> lock(mtx);

	static hpx::future<void> barrier(hpx::make_ready_future<void>());
	GET(barrier);
	nsteps = node_registry::begin()->second->get_step_num();
	timestamp = time(NULL);
	steps_elapsed = nsteps - start_step;
	time_elapsed = time(NULL) - start_time;
	start_time = timestamp;
	start_step = nsteps;
	std::vector<hpx::future<void>> futs1;
	for (auto& id : localities) {
		futs1.push_back(hpx::async<output_stage1_action>(id, fname, cycle));
	}
	GET(hpx::when_all(futs1));

	barrier = hpx::async([cycle,&fname]() {
		std::vector<hpx::future<std::vector<node_location::node_id>>> id_futs;
		for (auto& id : localities) {
			id_futs.push_back(hpx::async<output_stage2_action>(id, fname, cycle));
		}
		loc_ids.clear();
		for (auto& f : id_futs) {
			std::vector<node_location::node_id> these_ids = GET(f);
			for (auto& i : these_ids) {
				loc_ids.push_back(i);
			}
		}
		GET(hpx::async<output_stage3_action>(localities[0], fname, cycle));
	});

	if (block) {
		GET(barrier);
		barrier = hpx::make_ready_future<void>();
	}
}

void
local_load(const std::string&, std::vector<node_location::node_id> node_ids);
void
all_boundaries();

HPX_PLAIN_ACTION(local_load, local_load_action);
HPX_PLAIN_ACTION(all_boundaries, all_boundaries_action);

void all_boundaries() {
	std::vector<hpx::future<void>> futs;
	for (auto i = node_registry::begin(); i != node_registry::end(); i++) {
		futs.push_back(hpx::async([i]()
		{
			i->second->all_hydro_bounds();
		}));
	}
	GET(hpx::when_all(std::move(futs)));
}

void local_load(const std::string& fname, std::vector<node_location::node_id> node_ids) {

	std::vector<hpx::future<void>> futs;
	const auto me = hpx::find_here();
	static const auto names = grid::get_field_names();
	static const auto hydro_names = grid::get_hydro_field_names();
	for (const auto& i : node_ids) {
		node_location l;
		l.from_id(i);
		futs.push_back(hpx::new_<node_server>(me, l).then([&me,l](hpx::future<hpx::id_type>&& f)
		{
			const auto id = GET(f);
			auto client = node_client(id);
			load_registry::put(l.to_id(), id);
			const auto pid = load_registry::get(l.get_parent().to_id());
			node_client p(pid);
			GET(p.notify_parent(l,id));
		}));
	}

	for (int i = 0; i < futs.size(); i++) {
		GET(futs[i]);
	}

	GET(
			hpx::threads::run_as_os_thread([&fname]() {
		read_silo_var<integer> ri;
		read_silo_var<real> rr;
		DBfile* db = DBOpenReal( fname.c_str(), DB_PDB, DB_READ);
		load_options_from_silo(fname,db);
		const real dtime = rr(db, "time");
		const real rtime = rr(db, "rotational_time");
		const real grams = rr(db, "g");
		const real cm = rr(db, "cm");
		const real s = rr(db, "s");
		const real K = rr(db, "K");
		set_units(grams,cm,s,K);
		for( auto iter = node_registry::begin(); iter != node_registry::end(); iter++ ) {
			auto* node_ptr = iter->second;
			if( !node_ptr->refined() ) {
				const auto l = iter->first;
				grid& g = node_ptr->get_hydro_grid();
				const auto suffix = std::to_string(l.to_id());
				assert(db);
				std::vector<std::pair<std::string,real>> outflow;
				for( auto n : names ) {
					const auto name = n + std::string( "_") + suffix;
					const auto quadvar = DBGetQuadvar(db,name.c_str());
					g.set(n, static_cast<real*>(*(quadvar->vals)));
					DBFreeQuadvar(quadvar);
				}
				for( auto n : hydro_names ) {
					const auto name = n + std::string( "_") + suffix;
					std::string name_outflow = n + std::string("_outflow_") + suffix;
					const real o = read_silo_var<real>()(db, name_outflow.c_str());
					outflow.push_back(std::make_pair(n, o));
				}
				g.set_outflows(std::move(outflow));
			} node_ptr->set_time(dtime, rtime);
		}
		DBClose( db);
	}));
	if (hpx::get_locality_id() == 0) {
		grid::set_omega(opts.omega);
	}

}

void load_options_from_silo(std::string fname, DBfile* db) {
	const auto func = [&fname,&db]()
	{
		bool leaveopen;
		if( db == NULL )
		{
			db = DBOpenReal( fname.c_str(), DB_PDB, DB_READ);
			leaveopen = false;
		}
		else
		{
			leaveopen = true;
		}
		if (db != NULL)
		{
			read_silo_var<integer> ri;
			read_silo_var<real> rr;
			opts.n_species = ri(db, "n_species");
			opts.eos = eos_type(ri(db, "eos"));
			opts.gravity = ri(db, "gravity");
			opts.hydro = ri(db, "hydro");
			opts.omega = rr(db, "omega");
			opts.output_dt = rr(db, "output_frequency");
			opts.problem = problem_type(ri(db, "problem"));
			opts.radiation = ri(db, "radiation");
			opts.refinement_floor = rr(db, "refinement_floor");
			opts.xscale = rr(db, "xscale");
			if( !leaveopen)
			{
				DBClose(db);
			}
		}
		else
		{
			std::cout << "Could not load " << fname;
			throw;
		}
	};
	if (db == NULL) {
		GET(hpx::threads::run_as_os_thread(func));
	} else {
		func();
	}

}

hpx::id_type load_data_from_silo(std::string fname, node_server* root_ptr, hpx::id_type root) {
	load_registry::put(1, root);
	static int sz = localities.size();
	DBfile* db = GET(hpx::threads::run_as_os_thread(DBOpenReal, fname.c_str(), DB_PDB, DB_READ));
	epoch = GET(hpx::threads::run_as_os_thread(read_silo_var<integer>(), db, "epoch"));
	epoch++;
	if (db != NULL) {
		DBmultimesh* master_mesh = GET(hpx::threads::run_as_os_thread([&]()
		{
			return DBGetMultimesh( db, "mesh");
		}));
		GET(hpx::threads::run_as_os_thread(DBClose, db));
		std::vector<node_location::node_id> work;
		std::vector<hpx::future<void>> futs;
		const int chunk_size = master_mesh->nblocks / sz;
		for (int i = 0; i < master_mesh->nblocks; i++) {
			work.push_back(std::stoi(master_mesh->meshnames[i]));
			const int this_id = i / chunk_size;
			const int next_id = (i + 1) / chunk_size;
			assert(this_id < localities.size());
			if (this_id != next_id) {
				futs.push_back(hpx::async<local_load_action>(localities[this_id], fname, work));
				work.clear();
			}
		}
		GET(hpx::threads::run_as_os_thread(DBFreeMultimesh, master_mesh));
		for (auto& f : futs) {
			GET(f);
		}
	} else {
		std::cout << "Could not load " << fname;
		throw;
	}
	hpx::id_type rc = load_registry::get(1);
	load_registry::destroy();
	root_ptr->form_tree(root);
	GET(hpx::lcos::broadcast<all_boundaries_action>(localities));
	return std::move(rc);
}

silo_var_t::silo_var_t(const std::string& name, std::size_t nx) :
		name_(name), data_(nx*nx*nx) {
}

double&
silo_var_t::operator()(int i) {
	return data_[i];
}

double silo_var_t::operator()(int i) const {
	return data_[i];
}
