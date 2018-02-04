"""
Examples from HW sheet
Problems from website
"""


class Problem:
    def __init__(self, probToState, valueEstimates, rewards, solution=1):
        self.probToState = probToState
        self.valueEstimates = valueEstimates
        self.rewards = rewards
        self.solution = solution
        self.test = self.solution != 1


# Example from Piazza
problems = [
    # Piazza @126 Example
    # Problem(probToState=0.5,
    #         valueEstimates=[0, 3, 8, 2, 1, 2, 0],
    #         rewards=[0, 0, 0, 4, 1, 1, 1],
    #         solution=0.403032),

    # HW Examples
    # Problem(probToState=0.81,
    #         valueEstimates=[0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
    #         rewards=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
    #         solution=0.6226326309908364),
    # Problem(probToState=0.22,
    #         valueEstimates=[0.0, -5.2, 0.0, 25.4, 10.6, 9.2, 12.3],
    #         rewards=[-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1],
    #         solution=0.49567093118984556),
    # Problem(probToState=0.64,
    #         valueEstimates=[0.0, 4.9, 7.8, -2.3, 25.5, -10.2, -6.5],
    #         rewards=[-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9],
    #         solution=0.20550275877409016),

    # RLDM Problems
    Problem(probToState=0.4,
            valueEstimates=[0.0, -1.6, -2.2, 18.6, 0.0, 11.4, 0.0],
            rewards=[1.6, 7.1, 5.2, 5.9, 5.4, -2.7, -3.6]),
    Problem(probToState=1.0,
            valueEstimates=[0.0, 0.0, 0.0, 8.8, 0.0, 6.0, -1.7],
            rewards=[3.6, 1.1, 9.7, 3.4, -1.8, -3.6, 1.4]),
    Problem(probToState=0.31,
            valueEstimates=[0.0, -4.5, 4.9, 0.0, 17.8, 0.0, 9.6],
            rewards=[6.2, -3.4, 0.6, -3.2, 8.0, -0.5, -3.3]),
    Problem(probToState=1.0,
            valueEstimates=[0.0, 0.0, 19.2, 0.0, 11.9, 5.6, 20.5],
            rewards=[-4.5, 8.2, 4.3, 0.0, -2.9, 1.5, -0.8]),
    Problem(probToState=1.0,
            valueEstimates=[0.0, 6.8, 6.5, 22.3, 20.3, 9.2, 0.0],
            rewards=[3.5, 0.0, -0.8, -0.5, 3.2, 9.4, -3.9]),
    Problem(probToState=0.9,
            valueEstimates=[0.0, 5.1, 9.3, 0.0, 23.1, 0.0, 0.0],
            rewards=[-4.4, 4.1, 2.5, 9.8, 0.0, 4.5, 1.1]),
    Problem(probToState=0.96,
            valueEstimates=[0.0, 23.9, 0.0, 0.0, 6.6, -3.5, -0.1],
            rewards=[6.9, 0.0, -3.4, -3.1, 6.7, 6.2, 7.0]),
    Problem(probToState=0.45,
            valueEstimates=[0.0, 9.9, 0.0, 9.7, -1.0, -1.4, 15.2],
            rewards=[-3.5, -4.7, -1.1, 3.9, 0.0, 2.1, 0.8]),
    Problem(probToState=0.08,
            valueEstimates=[0.0, 14.5, 3.8, 11.0, 0.0, 5.5, 0.0],
            rewards=[-4.1, 6.2, -1.8, -4.7, 7.5, 4.8, -2.8]),
    Problem(probToState=0.24,
            valueEstimates=[0.0, 0.0, 13.4, 19.0, 0.0, 24.1, 0.0],
            rewards=[-2.6, -4.4, 0.8, 5.1, -2.3, 0.5, 8.5]),
]
