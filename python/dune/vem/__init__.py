from dune.vem._vem import *
registry = dict()
registry["space"] = {
        "AgglomeratedDG" : agglomerateddg,
        "AgglomeratedVEM" : agglomeratedvem
    }
