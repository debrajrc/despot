#ifndef DRN_H
#define DRN_H

#include <despot/interface/pomdp.h>
#include <despot/core/mdp.h>

namespace despot {

/* =============================================================================
 * SimpleState class
 * =============================================================================*/

class SimpleState: public State {
public:
    int id;

    SimpleState();
    SimpleState(int _id) :
            id(_id) {
    }

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

    std::map<int, SimpleState> states;  // state id -> state
    std::map<int, std::string> actionIdtoName; // action id -> action name
    std::map<std::string, int> actionNametoId; // action name -> action id
    std::map<int, std::map<int, std::map<int, double>>> transitions; // state id -> action id -> next state id -> probability
    std::map<int, int> obsMap; // state id -> observation id
    std::map<int, double> stateRewards; // state id -> reward
    std::map<int, std::map<int, double>> stateActionRewards; // state id -> action id -> reward
    double maxReward = 0;
    double minReward = 0;

    void loadDRNFile (std::string filename);

//public:
//	enum { // action
//		A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3
//		// A_WEST = 0, A_EAST = 1, A_NORTH = 2, A_SOUTH = 3, A_PLACE = 4
//	};
//	enum { // observation
//		O_INIT = 0, O_INGRID = 1, O_TARGET = 2, O_BAD = 3
//	};
//	enum { // rover position
//		P00 = 0, P01 = 1, P02 = 2, P03 = 3, P10 = 4, P11 = 5, P12 = 6, P13 = 7, P20 = 8, P21 = 9, P22 = 10, P23 = 11, P30 = 12, P31 = 13, P32 = 14, P33 = 15, PINIT = 16
//	};


//    struct DrnState {
//        int id;
//        std::map<std::string, DrnAction> actions;
//    };

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
//	ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",
//		std::string particle_bound_name = "DEFAULT") const;
	ValuedAction GetBestAction() const;
//	ScenarioLowerBound* CreateScenarioLowerBound(std::string name = "DEFAULT",
//		std::string particle_bound_name = "DEFAULT") const;

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
