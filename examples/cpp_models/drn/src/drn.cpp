#include "drn.h"
#include "despot/util/coord.h"

#include <regex>
#include <unordered_set>
//#include<limits>

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


/* =============================================================================
 * drn class
 * =============================================================================*/

Drn::Drn(const std::string& filename) {
    // std::cout << "loading drn file " << filename << std::endl;
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
    int& distToTarget = simple_state.distToTarget;
    int& distToBad = simple_state.distToBad;
    int actionId = action;

//    cout << "State: " << stateId << endl;
//    cout << " Action: " << actionIdtoName.at(actionId) << endl;

    if (transitions.at(stateId).find(actionId) == transitions.at(stateId).end()) {
        reward = minReward; // actions that are not allowed are bad
        obs = obsMap.at(stateId);
        return true;
    }

//    reward += stateRewards.at(stateId) + stateActionRewards.at(stateId).at(actionId); // predefined rewards in MDP, we may ignore them
    if (stateLabels.find(stateId) != stateLabels.end()) {
        for (const auto& label : stateLabels.at(stateId)) {
            if (label == "bad") {
                obs = obsMap.at(stateId);
                reward = minReward; // This where the bad reward is, need to change if needed
//                cout << "Bad state reached in " << counter << " steps" << endl;
                return true;
            }
            if (label == "goal") {
                obs = obsMap.at(stateId);
                reward = maxReward; // This where the good reward is, need to change if needed
//                cout << "Goal reached in " << counter << " steps" << endl;
                return true;
            }
        }
    }

    auto currentTransitions = transitions.at(stateId).at(actionId);
    bool stateUpdated = false;
    double cumulativeProbability = 0;
    assert (!currentTransitions.empty());
    for (auto nextState : currentTransitions) {
        cumulativeProbability += nextState.second;
//        cout << "Cumulative probability: " << cumulativeProbability << endl;
        if (rand_num <= cumulativeProbability) {
            stateId = nextState.first;
            distToTarget = distanceToTarget.at(stateId);
            distToBad = distanceToBad.at(stateId);
            stateUpdated = true;
            break;
        }
    }

    assert (stateUpdated);

    obs = obsMap.at(stateId);
    return false;
};

void Drn::CalculateDistances() {
    cout << "Calculating distances to target and bad states" << endl;
    // Initialize queue
        std::queue<std::tuple<int, int, bool>> queue; // (state, distance, is_from_target)
//        std::unordered_set<int> visitedTarget;
//        std::unordered_set<int> visitedBad;

    // Initialize distances to infinity
    for (const auto& stateReward : stateRewards) {
        int state = stateReward.first;
        cout << "State: " << state << endl;
        if (stateLabels.find(state) != stateLabels.end()) {
            bool badFound = false;
            bool targetFound = false;
            for (const auto& label : stateLabels.at(state)) {
                if (label == "bad") {
                    distanceToBad[state] = 0;
                    queue.emplace(state, 0, false);
                    cout << "Adding state " << state << " to bad queue with distance " << 0 << endl;
                    badFound = true;
//                    visitedBad.insert(state);
                }
                if (label == "goal") {
                    distanceToTarget[state] = 0;
                    queue.emplace(state, 0, true);
                    cout << "Adding state " << state << " to target queue with distance " << 0 << endl;
                    targetFound = true;
//                    visitedTarget.insert(state);
                }
                if (!badFound){
                    distanceToBad[state] = INF;
                    cout << "State " << state << " is not bad, setting distance to bad to " << INF << endl;
                }
                if (!targetFound){
                    distanceToTarget[state] = INF;
                    cout << "State " << state << " is not target, setting distance to target to " << INF << endl;
                }
            }
        } else {
            distanceToBad[state] = INF;
            cout << "State " << state << " is not bad, setting distance to bad to " << INF << endl;
            distanceToTarget[state] = INF;
            cout << "State " << state << " is not target, setting distance to target to " << INF << endl;
        }
    }

    // print the initial distances
    cout << "Initial distances to target and bad states" << endl;
    cout << "State | Distance to Target | Distance to Bad" << endl;
    for (const auto& state : distanceToTarget) {
        cout << state.first << " | " << state.second << " | " << distanceToBad[state.first] << endl;
    }

    // multi-target BFS to calculate distances
    while (!queue.empty()) {
        auto [state, distance, is_from_target] = queue.front();
        cout << "State: " << state << " Distance: " << distance << " Is from target: " << is_from_target << endl;
        queue.pop();

//        if (is_from_target) {
//            if (distanceToTarget[state] < distance) continue;
////            distanceToTarget[state] = distance;
//        } else {
//            if (distanceToBad[state] < distance) continue;
////            distanceToBad[state] = distance;
//        }

        if (reversedGraph.find(state) != reversedGraph.end()) {
            for (const auto& prevState : reversedGraph[state]) {
                if (is_from_target && distance + 1 < distanceToTarget[prevState]) {
                    queue.emplace(prevState, distance + 1, is_from_target);
//                    visitedTarget.insert(prevState);
                    distanceToTarget [prevState] = distance + 1;
                    cout << "Adding state " << prevState << " to target queue with distance " << distance + 1 << endl;
                } else if (!is_from_target && distance + 1 < distanceToBad[prevState]) {
                    queue.emplace(prevState, distance + 1, is_from_target);
                    distanceToBad [prevState] = distance + 1;
                    cout << "Adding state " << prevState << " to bad queue with distance " << distance + 1 << endl;
                }
            }
        }
    }

    // debugging print
    cout << "State | Distance to Target | Distance to Bad" << endl;
    for (const auto& state : distanceToTarget) {
        cout << state.first << " | " << state.second << " | " << distanceToBad[state.first] << endl;
    }
}

void Drn::loadDRNFile(const std::string filename) {
//    SimpleState* currentState = nullptr;
    int currentState = -1;
    int currentActionId = 0;
//    double lastStateReward = 0;
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

        if (token == "state") { // state 0 {6} [0] bad
            std::regex re(R"(state\s+(\d+)\s+\{(\d+)\}\s*(?:\[(\d+)\])?\s*([\s\w]+)?)");
            std::smatch match;

            if (std::regex_search(line, match, re)) {
                int stateId = std::stoi(match[1]);
                int obs = std::stoi(match[2]);
//                states[stateId] = SimpleState(stateId);
                currentState = stateId;// &states[stateId];
                double stateReward = std::stod(match[3].matched ? match[3].str() : "0");
                stateRewards[stateId] = stateReward;
//                lastStateReward = stateReward;
                obsMap[stateId] = obs;
                lastState = stateId;
                std::string stateLabel = match[4].matched ? match[4].str() : "";
                std::vector<std::string> stateLabelList;
                std::istringstream stateLabelStream(stateLabel);
                std::string word;
                while (stateLabelStream >> word) {
                    stateLabelList.push_back(word);
                }
                std::string stateLabelText;
                if (!stateLabel.empty()) {
                    stateLabels[stateId] = stateLabelList;
                    stateLabelText = " and label "; // + stateLabel;
                    for (const auto& label : stateLabelList) {
                        stateLabelText += label + " ";
                    }
                } else {
                    stateLabelText = "";
                }
//                cout << "State " << stateId << " with reward " << stateReward << " observation " << obs << stateLabelText << endl;
            } else {
                cout << "Error parsing state line: " << line << endl;
            }
        } else if (token == "action") { // action (init) [0] //  && currentState != nullptr
            std::regex re(R"(action\s+([\(\)\w]+)\s*(?:\[([\-\d]+)\])?)");
            std::smatch match;
            if (std::regex_search(line, match, re)) {
                std::string actionName = match[1];
                if (actionName.front() == '(') {
                    actionName = actionName.substr(1, actionName.size() - 2);
                }
                double reward = std::stod(match[2].matched ? match[2].str() : "0");
                auto actionIdIter = actionNametoId.find(actionName);
                int actionId;
                if (actionIdIter == actionNametoId.end()) {
                    actionNametoId[actionName] = currentActionId;
                    actionIdtoName[currentActionId] = actionName;
//                     cout << "Action " << actionName << " with id " << currentActionId << endl;
                    actionId = currentActionId;
                    currentActionId++;
                } else {
                    actionId = actionIdIter->second;
                }
                stateActionRewards[currentState][actionId] = reward;
//                maxReward = 1;//std::max(maxReward, reward+lastStateReward);
//                minReward = 0;//std::min(minReward, reward+lastStateReward);
                lastAction = actionId;
//                 cout << "Action " << actionName << " with reward " << reward << endl;
            } else {
//                cout << "Error parsing action line: " << line << endl;
            }
        } else if ((line.find(':') != std::string::npos) && lastState != -2 && lastAction != -2) {  // 1 : 0.111111
            std::istringstream transitionStream(line);
            size_t pos = line.find(':');
            std::string stateStr = line.substr(0, pos);
            std::string probStr = line.substr(pos + 1);
            int nextState = std::stoi(stateStr);
            double probability = std::stod(probStr);
            transitions[lastState][lastAction][nextState] = probability;

            // I am creating a reversed graph to create a handmade reward function
            if (reversedGraph.find(nextState) == reversedGraph.end()) {
                reversedGraph[nextState] = std::vector<int>();
            }
            reversedGraph[nextState].push_back(lastState);

//             cout << "Transition from" << lastState << " to state " << nextState << " with action " << lastAction << " with probability " << probability << endl;
        }
    }
    // cout << "file read" << endl;
    // print actionIdtoName
     for (const auto& action : actionIdtoName) {
         cout << action.first << " " << action.second << endl;
     }

    CalculateDistances();
}

/* ================================================
 * Functions related to beliefs and starting states
 * ================================================*/

double Drn::ObsProb(OBS_TYPE obs, const State& state,
	ACT_TYPE action) const {
	const SimpleState& simple_state = static_cast<const SimpleState&>(state);
    int stateId = simple_state.id;
    return int(obs) == obsMap.at(stateId);
}

State* Drn::CreateStartState(string type) const {
	return new SimpleState(0, distanceToTarget.at(0), distanceToBad.at(0)); // TODO: maybe I should look for the label init in the file, but in all the examples, 0 is the initial state
}

Belief* Drn::InitialBelief(const State* start, string type) const {
	if (type == "DEFAULT" || type == "PARTICLE") {
		vector<State*> particles;

		SimpleState* s = static_cast<SimpleState*>(Allocate(-1, 1));
		s->id = 0; // TODO: maybe I should look for the label init in the file, but in all the examples, 0 is the initial state
        s->distToTarget = distanceToTarget.at(0);
        s->distToBad = distanceToBad.at(0);
        particles.push_back(s);

		return new ParticleBelief(particles, this);
	} else {
		cerr << "[drn::InitialBelief] Unsupported belief type: " << type << endl;
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

ValuedAction Drn::GetBestAction() const {
	return ValuedAction(0, minReward);
}

class DistBasedParticleLowerBound: public ParticleLowerBound {
protected:
    const Drn* drn_;
public:
    DistBasedParticleLowerBound(const DSPOMDP* model) :
            ParticleLowerBound(model),
            drn_(static_cast<const Drn*>(model)) {
    }

    ValuedAction Value(const vector<State*>& particles) const {
        const SimpleState& state = static_cast<const SimpleState&>(*particles[0]);
        double badScore = 0;
        if (state.distToBad == INF) {
            badScore = 0;
        } else {
            badScore = 1.0/(state.distToBad+1);
        }
        double targetScore = 0;
        if (state.distToTarget == INF) {
            targetScore = 0;
        } else {
            targetScore = 1.0/(state.distToTarget+1);
        }
        return ValuedAction(0, State::Weight(particles) * (targetScore));
    }
};

class StupidParticleLowerBound: public ParticleLowerBound {

public:
    StupidParticleLowerBound(const DSPOMDP* model) :
        ParticleLowerBound(model) {
    }

    ValuedAction Value(const vector<State*>& particles) const {
        return ValuedAction(0, -1);
    }
};

ScenarioLowerBound* Drn::CreateScenarioLowerBound(string name, string particle_bound_name) const {
    return new StupidParticleLowerBound(this);
//    return new DistBasedParticleLowerBound(this);
}

//* =================
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

	out << "\tState id = " << simple_state.id << "\n\tdistToTarget = " << simple_state.distToTarget << "\n\tdistToBad = " << simple_state.distToBad << endl;
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
