import numpy as np

from hw5.problems import sample_problems, rldm_problems

NO_FIGHT = 0
FIGHT = 1
UNKNOWN = -1


class KWIKFightDetector:
    def __init__(self, max_unknown):
        # Agent memory fields based on data from first prediction
        self.initialized = False
        self.memory = None

        # Label request upper bound
        self.max_unknown = max_unknown

        # Track discovered instigator and peacemaker patron indices
        self.possible_instigators = []
        self.possible_peacemakers = []
        self.known_instigator = False
        self.known_peacemaker = False

    def _setup(self, num_patrons):
        self.initialized = True

        # Remember labeled training data
        self.memory = np.empty((0, num_patrons + 1))

        # Track discovered instigator and peacemaker patron indices
        self.possible_instigators = list(range(num_patrons))
        self.possible_peacemakers = list(range(num_patrons))

    def predict(self, patrons):
        """
        Predict outcome of an episode

         0 = NO FIGHT
         1 = FIGHT
        -1 = I DON'T KNOW

        :param patrons: boolean list of patrons present
        :return:
        """
        # First update suspect list
        if not self.initialized:
            self._setup(len(patrons))

        # Logic based on current episode information
        if all(patrons) or all(np.invert(patrons)):
            return NO_FIGHT

        # Convert to list of indices ("patron IDs")
        patron_ids = np.asarray(np.where(patrons)).flatten()

        if self.known_peacemaker is not False:

            # Peacemaker is present
            if self.known_peacemaker in patron_ids:
                return NO_FIGHT

            if self.known_instigator is not False:

                # Instigator present and peacemaker is not
                if self.known_instigator in patron_ids:
                    return FIGHT

                # Instigator not present
                return NO_FIGHT

        if self.known_instigator is not False:

            # Instigator is not present
            if self.known_instigator not in patron_ids:
                return NO_FIGHT

        # Check memory if we've seen this before
        for m in self.memory:
            if np.all(m[:-1] == patrons):
                return int(m[-1])

        # No immediate information and insufficient historical data
        return UNKNOWN

    def train(self, patrons, fight):
        learned_instigator = False
        learned_peacemaker = False

        if fight:
            # If fight occurred, we know instigator was present and
            # peacemaker is absent
            self.possible_instigators = np.intersect1d(self.possible_instigators, np.where(patrons))
            self.possible_peacemakers = np.intersect1d(self.possible_peacemakers, np.where(np.invert(patrons)))

        else:
            # If no fight occurred, we know (instigator was not present) OR (instigator AND peacemaker are present)
            if self.known_instigator is not False and self.known_instigator in np.where(patrons):
                self.possible_peacemakers = np.intersect1d(self.possible_peacemakers, np.where(patrons))

        # Save to memory
        record = np.append(patrons, fight)
        self.memory = np.vstack([self.memory, record])

        # Narrow down suspects to 1
        if len(self.possible_instigators) == 1:
            self.known_instigator = self.possible_instigators[0]
            learned_instigator = True
        if len(self.possible_peacemakers) == 1:
            self.known_peacemaker = self.possible_peacemakers[0]
            learned_peacemaker = True

        return learned_instigator, learned_peacemaker


if __name__ == '__main__':

    for p, problem in enumerate(rldm_problems):

        problem_predictions = []
        all_patrons = problem['patrons']
        all_fights = problem['fight']
        num_episodes = len(all_patrons)

        agent = KWIKFightDetector(len(all_patrons[0]))

        print('*' * 80)
        print('Problem {}'.format(p + 1))
        print('{} episodes'.format(num_episodes))
        print('{} patrons'.format(len(all_patrons[0])))

        assert len(all_patrons) == len(all_fights)

        for e in range(num_episodes):
            patron_list = np.array(all_patrons[e], dtype=bool)
            fight_occurred = np.array(all_fights[e], dtype=bool)
            prediction = agent.predict(patron_list)
            # print('Episode {}: {}'.format(e, prediction))
            problem_predictions.append(prediction)
            if prediction == UNKNOWN:
                # print('Agent cannot predict: training agent with labeled episode data...')
                instigator, peacemaker = agent.train(patron_list, fight_occurred)

                # Check training information gain
                if instigator:
                    print('Identified instigator after {} episodes'.format(e + 1))
                if peacemaker:
                    print('Identified peacemaker after {} episodes'.format(e + 1))

        print('Solution:\n{}'.format(problem_predictions))
        print()
