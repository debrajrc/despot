#ifndef DRN_H
#define DRN_H

#include <despot/interface/pomdp.h>
#include <despot/core/mdp.h>

#include <limits>

namespace despot {

const int INF = std::numeric_limits<int>::max(); // infinity

    /* =============================================================================
 * SimpleState class
 * =============================================================================*/

class SimpleState: public State {
public:
    int id;
    int distToTarget;
    int distToBad;

    SimpleState();
    SimpleState(int _id, int _distToTarget, int _distToBad) : id(_id), distToTarget(_distToTarget), distToBad(_distToBad) {}

    ~SimpleState();
};
/* =============================================================================
 * GridAvoid class
 * =============================================================================*/

class Drn: public DSPOMDP {
protected:
	mutable MemoryPool<SimpleState> memory_pool_;

//	std::vector<State*> states_;
//
//	mutable std::vector<ValuedAction> mdp_policy_;

//public:
//    struct DrnTransition {
//        int nextState;
//        double probability;
//    };

//    struct DrnAction {
//        std::string name;
//        double reward;
//        std::vector<DrnTransition> transitions;
//    };

    private:

//    std::map<int, SimpleState> states;  // state id -> state
    std::map<int, std::string> actionIdtoName; // action id -> action name
    std::map<std::string, int> actionNametoId; // action name -> action id
    std::map<int, std::map<int, std::map<int, double>>> transitions; // state id -> action id -> next state id -> probability
    std::map<int, int> obsMap; // state id -> observation id
    std::map<int, double> stateRewards; // state id -> reward
    std::map<int, std::map<int, double>> stateActionRewards; // state id -> action id -> reward
    std::map<int, std::vector<std::string>> stateLabels; // state id -> vector of labels
    std::map<int, int> distanceToTarget; // state id -> distance to target
    std::map<int, int> distanceToBad; // state id -> distance to bad state
    const double maxReward = 1; // maximum reward is when we reach the target
    const double minReward = -1; // minimum reward is when we reach a bad state
    std::map<int,std::vector<int>> reversedGraph; // state id -> vector of state ids
    // const int maxCounter = 10; // maximum counter value; old code, not used anymore
//    const int alpha = 1; // alpha value for the reward function
//    const int beta = 1; // beta value for the reward function


    void CalculateDistances(); // calculate distances to target and bad states, need to run it after loadDRNFile
    void loadDRNFile (std::string filename); // load the DRN file and parse it

public:
	Drn(const std::string& filename);

	/* Returns total number of actions.*/
	int NumActions() const;

	/* Deterministic simulative model.*/
	bool Step(State& state, double rand_num, ACT_TYPE action, double& reward,
		OBS_TYPE& obs) const;

	/* Functions related to beliefs and starting states.*/
	double ObsProb(OBS_TYPE obs, const State& state, ACT_TYPE action) const;
	State* CreateStartState(std::string type = "PARTICLE") const;
	Belief* InitialBelief(const State* start, std::string type = "PARTICLE") const;

	/* Bound-related functions.*/
	double GetMaxReward() const;
	ValuedAction GetBestAction() const;

	/* Memory management.*/
	State* Allocate(int state_id, double weight) const;
	State* Copy(const State* particle) const;
	void Free(State* particle) const;
	int NumActiveParticles() const;

	/* Display.*/
	void PrintState(const State& state, std::ostream& out = std::cout) const;
	void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;
	void PrintObs(const State& state, OBS_TYPE observation,
		std::ostream& out = std::cout) const;
	void PrintAction(ACT_TYPE action, std::ostream& out = std::cout) const;

    ScenarioLowerBound *CreateScenarioLowerBound(std::string name, std::string particle_bound_name) const;

//        ScenarioUpperBound *CreateScenarioUpperBound(std::string name, std::string particle_bound_name) const;
    };

} // namespace despot

#endif
