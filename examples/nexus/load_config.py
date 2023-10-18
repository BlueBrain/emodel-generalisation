import json
from pathlib import Path

from emodel_generalisation.model.access_point import AccessPoint

if __name__ == "__main__":
    config = json.load(open("config.json"))

    access_point = AccessPoint(nexus_config='config.json', emodel_dir='config')
    emodel = 'ACAd1_L1_DAC_bNAC_0'
    print(access_point.get_recipes(emodel))
    print(access_point.final[emodel])
    print(access_point.get_settings(emodel))
