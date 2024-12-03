#include "drn.h"

#include <algorithm>

#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/core/particle_belief.h>

using namespace std;

namespace despot {

/* =============================================================================
 * SimpleState class
 * =============================================================================*/

    SimpleState::SimpleState() {
    }

    SimpleState::~SimpleState() {
    }

//string SimpleState::text() const {
//	int x = rover_position / 4;
//	int y = rover_position % 4;
//	return to_string(x) + ", " + to_string(y);
//}

/* =============================================================================
 * SimpleRockSample class
 * =============================================================================*/

Drn::Drn(const std::string& filename) {
        loadDRNFile(filename);
}

/* ======
 * Action
 * ======*/

int Drn::NumActions() const {
	return actionNametoId.size();
}

/* ==============================
 * simulative model
 * ==============================*/


bool Drn::Step(State& state, double rand_num, ACT_TYPE action,
        double& reward, OBS_TYPE& obs) const {
    SimpleState& simple_state = static_cast<SimpleState&>(state);
    int& stateId = simple_state.id;
    int actionId = action;

//    cout << "State: " << stateId << endl;
//    cout << " Action: " << actionIdtoName.at(actionId) << endl;

    if (transitions.at(stateId).find(actionId) == transitions.at(stateId).end()) {
//        cout << "No transitions for" << stateId << " " << actionIdtoName.at(actionId) << endl;
        reward -= 1000;
        return true;
    }

    reward += stateRewards.at(stateId) + stateActionRewards.at(stateId).at(actionId);

    auto currentTransitions = transitions.at(stateId).at(actionId);
    double cumulativeProbability = 0;
    for (auto nextState : currentTransitions) {
        cumulativeProbability += nextState.second;
        if (rand_num <= cumulativeProbability) {
            stateId = nextState.first;
            break;
        }
    }

    obs = obsMap.at(stateId);

    return false;

};

void Drn::loadDRNFile(const std::string filename) {
    SimpleState* currentState = nullptr;
    int currentActionId = 0;
    double lastStateReward = 0;
    int lastState = -2;
    int lastAction = -2;

    cout << "Loading DRN file " << filename << endl;
//    cout << line << endl;

    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "state") { // state 0 {6} [0]
            int stateId;
            std::string obsWithBrackets, stateRewardWithBrackets;
            iss >> stateId >> obsWithBrackets >> stateRewardWithBrackets;
            states[stateId] = SimpleState(stateId);
            currentState = &states[stateId];
            int obs = std::stoi(obsWithBrackets.substr(1, obsWithBrackets.size() - 2));
            double stateReward = std::stod(stateRewardWithBrackets.substr(1, stateRewardWithBrackets.size() - 2));
            stateRewards[stateId] = stateReward;
            lastStateReward = stateReward;
            obsMap[stateId] = obs;
            lastState = stateId;
            cout << "State " << stateId << " with reward " << stateReward << " and observation " << obs << endl;
        } else if (token == "action" && currentState != nullptr) { // action (init) [0]
            std::string actionNameWithBrackets, rewardWithBrackets;
            iss >> actionNameWithBrackets >> rewardWithBrackets;
            std::string actionName = actionNameWithBrackets.substr(1, actionNameWithBrackets.size() - 2);
            double reward = std::stod(rewardWithBrackets.substr(1, rewardWithBrackets.size() - 2));
            auto actionIdIter = actionNametoId.find(actionName);
            int actionId;
            if (actionIdIter == actionNametoId.end()) {
                actionNametoId[actionName] = currentActionId;
                actionIdtoName[currentActionId] = actionName;
                cout << "Action " << actionName << " with id " << currentActionId << endl;
                actionId = currentActionId;
                currentActionId++;
            } else {
                actionId = actionIdIter->second;
            }
            stateActionRewards[currentState->id][actionId] = reward;
            maxReward = std::max(maxReward, reward+lastStateReward);
            minReward = std::min(minReward, reward+lastStateReward);
            lastAction = actionId;
            cout << "Action " << actionName << " with reward " << reward << endl;

        } else if ((line.find(':') != std::string::npos) && lastState != -2 && lastAction != -2) {  // 1 : 0.111111
            std::istringstream transitionStream(line);
            size_t pos = line.find(':');
            std::string stateStr = line.substr(0, pos);
            std::string probStr = line.substr(pos + 1);
            int nextState = std::stoi(stateStr);
            double probability = std::stod(probStr);
            transitions[lastState][lastAction][nextState] = probability;
            cout << "Transition from" << lastState << " to state " << nextState << " with action " << lastAction << " with probability " << probability << endl;
        }
    }
    cout << "file read" << endl;
}

/* ================================================
 * Functions related to beliefs and starting states
 * ================================================*/

double Drn::ObsProb(OBS_TYPE obs, const State& state,
	ACT_TYPE action) const {
	const SimpleState& simple_state = static_cast<const SimpleState&>(state);
    int stateId = simple_state.id;
    return obs == obsMap.at(stateId);
}

State* Drn::CreateStartState(string type) const {
	return new SimpleState(0); // TODO: maybe I should look for the label init in the file, but in all the examples, 0 is the initial state
}

Belief* Drn::InitialBelief(const State* start, string type) const {
	if (type == "DEFAULT" || type == "PARTICLE") {
		vector<State*> particles;

		SimpleState* s = static_cast<SimpleState*>(Allocate(-1, 1));
		s->id = 0; // TODO: maybe I should look for the label init in the file, but in all the examples, 0 is the initial state
		particles.push_back(s);

		return new ParticleBelief(particles, this);
	} else {
		cerr << "[SimpleRockSample::InitialBelief] Unsupported belief type: " << type << endl;
		exit(1);
	}
}

/* ========================
 * Bound-related functions.
 * ========================*/
/*
Note: in the following bound-related functions, only GetMaxReward() and 
GetBestAction() functions are required to be implemented. The other 
functions (or classes) are for custom bounds. You don't need to write them
if you don't want to use your own custom bounds. However, it is highly 
recommended that you build the bounds based on the domain knowledge because
it often improves the performance. Read the tutorial for more details on how
to implement custom bounds.
*/
double Drn::GetMaxReward() const {
	return maxReward;
}

// class SimpleRockSampleParticleUpperBound: public ParticleUpperBound {
// protected:
// 	// upper_bounds_[pos][status]:
// 	//   max possible reward when rover_position = pos, and rock_status = status.
// 	vector<vector<double> > upper_bounds_;

// public:
// 	SimpleRockSampleParticleUpperBound(const DSPOMDP* model) {
// 		upper_bounds_.resize(3);
// 		upper_bounds_[0].push_back(Globals::Discount(1) * 10);
// 		upper_bounds_[0].push_back(10 + Globals::Discount(2) * 10);
// 		upper_bounds_[1].push_back(10);
// 		upper_bounds_[1].push_back(Globals::Discount(1) * 10 + Globals::Discount(3) * 10);
// 		if (upper_bounds_[1][1] < 10)
// 			upper_bounds_[1][1] = 10;
// 		upper_bounds_[2].push_back(0);
// 		upper_bounds_[2].push_back(0);
// 	}

// 	double Value(const State& s) const {
// 		const SimpleState& state = static_cast<const SimpleState&>(s);
// 		return upper_bounds_[state.rover_position][state.rock_status];
// 	}
// };

//ScenarioUpperBound* GridAvoid::CreateScenarioUpperBound(string name,
//	string particle_bound_name) const {
//	ScenarioUpperBound* bound = NULL;
//	bound = new TrivialParticleUpperBound(this);
//	return bound;
//}

ValuedAction Drn::GetBestAction() const {
	return ValuedAction(0, minReward);
}

//class DefaultEastPolicy: public DefaultPolicy {
//public:
//	enum { // action
//		A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3
//	};
//	DefaultEastPolicy(const DSPOMDP* model, ParticleLowerBound* bound) :
//		DefaultPolicy(model, bound) {
//	}
//
//	ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
//		History& history) const {
//		return A_EAST; // move east
//	}
//};
//class DefaultWestPolicy: public DefaultPolicy {
//public:
//	enum { // action
//		A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3
//	};
//	DefaultWestPolicy(const DSPOMDP* model, ParticleLowerBound* bound) :
//		DefaultPolicy(model, bound) {
//	}
//
//	ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
//		History& history) const {
//		return A_WEST; // move east
//	}
//};
//class DefaultNorthPolicy: public DefaultPolicy {
//public:
//	enum { // action
//		A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3
//	};
//	DefaultNorthPolicy(const DSPOMDP* model, ParticleLowerBound* bound) :
//		DefaultPolicy(model, bound) {
//	}
//
//	ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
//		History& history) const {
//		return A_NORTH; // move east
//	}
//};
//class DefaultSouthPolicy: public DefaultPolicy {
//public:
//	enum { // action
//		A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3
//	};
//	DefaultSouthPolicy(const DSPOMDP* model, ParticleLowerBound* bound) :
//		DefaultPolicy(model, bound) {
//	}
//
//	ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
//		History& history) const {
//		return A_SOUTH; // move east
//	}
//};
//
//ScenarioLowerBound* GridAvoid::CreateScenarioLowerBound(string name,
//	string particle_bound_name) const {
//	ScenarioLowerBound* bound = NULL;
//	if (name == "TRIVIAL" || name == "DEFAULT") {
//		bound = new TrivialParticleLowerBound(this);
//	} else if (name == "EAST") {
//		bound = new DefaultEastPolicy(this, CreateParticleLowerBound(particle_bound_name));
//	} else if (name == "WEST") {
//		bound = new DefaultWestPolicy(this, CreateParticleLowerBound(particle_bound_name));
//	} else if (name == "NORTH") {
//		bound = new DefaultNorthPolicy(this, CreateParticleLowerBound(particle_bound_name));
//	} else if (name == "SOUTH") {
//		bound = new DefaultSouthPolicy(this, CreateParticleLowerBound(particle_bound_name));
//	} else {
//		cerr << "Unsupported lower bound algorithm: " << name << endl;
//		exit(0);
//	}
//	// bound = new TrivialParticleLowerBound(this);
//	return bound;
//}
//

//class StupidParticleUpperBound: public ParticleUpperBound {
//
//public:
//    StupidParticleUpperBound(const despot::Drn* model) {
//    }
//
//    double Value(const despot::State& state) const {
//        return 0;
//    }
//};

class StupidParticleLowerBound: public ParticleLowerBound {

public:
    StupidParticleLowerBound(const DSPOMDP* model) :
        ParticleLowerBound(model) {
    }

    ValuedAction Value(const vector<State*>& particles) const {
        return ValuedAction(-1, 0);
    }
};

//class StupidParticleUpperBound: public ParticleUpperBound {
//
//public:
//    StupidParticleUpperBound(const DSPOMDP* model) {
//    }
//
//    double Value(const State& state) const {
//        return 0;
//    }
//};

ScenarioLowerBound* Drn::CreateScenarioLowerBound(string name, string particle_bound_name) const {
    return new StupidParticleLowerBound(this);
}

//ScenarioUpperBound* Drn::CreateScenarioUpperBound(string name, string particle_bound_name) const {
//    return new StupidParticleUpperBound(this);
//}


    ///* =================
// * Memory management
// * =================*/
//
State* Drn::Allocate(int state_id, double weight) const {
    SimpleState* state = memory_pool_.Allocate();
	state->state_id = state_id;
	state->weight = weight;
	return state;
}

State* Drn::Copy(const State* particle) const {
    SimpleState* state = memory_pool_.Allocate();
	*state = *static_cast<const SimpleState*>(particle);
	state->SetAllocated();
	return state;
}

void Drn::Free(State* particle) const {
	memory_pool_.Free(static_cast<SimpleState*>(particle));
}

int Drn::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

/* =======
 * Display
 * =======*/

void Drn::PrintState(const State& state, ostream& out) const {
	const SimpleState& simple_state = static_cast<const SimpleState&>(state);

	out << "State id = " << simple_state.id << ";"
		<< endl;
}

void Drn::PrintObs(const State& state, OBS_TYPE observation,
	ostream& out) const {
	out << observation << endl;
}

void Drn::PrintBelief(const Belief& belief, ostream& out) const {
	const vector<State*>& particles =
		static_cast<const ParticleBelief&>(belief).particles();

	vector<double> pos_probs(3);
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		const SimpleState* state = static_cast<const SimpleState*>(particle);
		pos_probs[state->id] += particle->weight;
	}

	for (int i = 0; i < 3; i++) {
        out << "Position " << i << ": " << pos_probs[i] << endl;
    }
}

void Drn::PrintAction(ACT_TYPE action, ostream& out) const {
    out << actionIdtoName.at(action) << endl;
}

} // namespace despot
