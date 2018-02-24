import json


def mdp_to_json(mdp_obj, save=False):
    if save:
        with open('problem.json', 'w') as outfile:
            json.dump(mdp_obj, outfile)
    else:
        json_data = json.dumps(mdp_obj)
        return json_data
