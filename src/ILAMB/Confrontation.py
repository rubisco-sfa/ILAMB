from CO2MaunaLoa import CO2MaunaLoa
from GPPFluxnetGlobalMTE import GPPFluxnetGlobalMTE

class Confrontation():
    """
    A class for managing confrontations
    """
    def __init__(self):
        c = {}

        # categories of confrontations
        c["EcosystemAndCarbonCycle"] = {}
        c["HydrologyCycle"]          = {}
        c["RadiationAndEnergyCycle"] = {}
        c["Forcings"]                = {}
        #c["EcosystemAndCarbonCycle"]["CO2"] = {"CO2MaunaLoa":CO2MaunaLoa}
        c["EcosystemAndCarbonCycle"]["GPP"] = {"GPPFluxnetGlobalMTE":GPPFluxnetGlobalMTE}
        
        self.confrontation = c

    def __str__(self):
        s  = "Confrontations\n"
        s += "--------------\n"
        for cat in self.confrontation.keys():
            s += "  |-> %s\n" % cat
            for area in self.confrontation[cat].keys():
                s += "        |-> %s\n" % area
                for obs in self.confrontation[cat][area].keys():
                    s += "              |-> %s\n" % obs
        return s

    def list(self):
        c = []
        for cat in self.confrontation.keys():
            for area in self.confrontation[cat].keys():
                for obs in self.confrontation[cat][area].keys():
                    try:
                        C = self.confrontation[cat][area][obs]
                        c.append(C())
                    except:
                        pass
        return c
