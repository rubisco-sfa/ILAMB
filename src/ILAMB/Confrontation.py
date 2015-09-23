from ILAMB.GenericConfrontation import GenericConfrontation
import os

class Confrontation():
    """
    A class for managing confrontations
    """
    def __init__(self,regions=["global.large"]):
    
        root = os.environ["ILAMB_ROOT"]+"/DATA"

        c = {}
        
        # Ecosystem and Carbon Cycle -----------------------------------------------
        e = {}
        e["GPP"] = []
        try:
            e["GPP"].append(GenericConfrontation("GPPFluxnetGlobalMTE",
                                                 root + "/gpp/FLUXNET-MTE/derived/gpp.nc",
                                                "gpp",
                                                 regions=regions,
                                                 cmap="Greens"))
        except:
            pass
        c["EcosystemAndCarbonCycle"] = e

        # Hydrology Cycle ----------------------------------------------------------
        h = {}
        h["LE"] = []
        try:
            h["LE"].append(GenericConfrontation("LEFluxnetGlobalMTE",
                                                root + "/le/FLUXNET-MTE/derived/le.nc",
                                                "hfls",
                                                alternate_vars=["le"],
                                                regions=regions,
                                                cmap="Oranges",
                                                land=True))
        except:
            pass
        try:
            h["LE"].append(GenericConfrontation("LEFluxnetSites",
                                                root + "/le/FLUXNET/derived/le.nc",
                                                "hfls",
                                                alternate_vars=["le"],
                                                regions=regions,
                                                cmap="Oranges"))
        except:
            pass
        
        c["HydrologyCycle"] = h
        
        # Radiation and Energy Cycle -----------------------------------------------

        # Forcings -----------------------------------------------------------------
        f = {}
        f["PR"] = []
        try:
            f["PR"].append(GenericConfrontation("PRGPCP2",
                                                root + "/pr/GPCP2/derived/pr.nc",
                                                "pr",
                                                regions=regions,
                                                cmap="Blues",
                                                land=True))
        except:
            pass

        c["Forcings"] = f
                               
        self.confrontation = c

    def __str__(self):
        
        s  = "Confrontations\n"
        s += "--------------\n"
        for cat in self.confrontation.keys():
            s += "  |-> %s\n" % cat
            for area in self.confrontation[cat].keys():
                s += "        |-> %s\n" % area
                for obs in self.confrontation[cat][area]:
                    s += "              |-> %s\n" % obs.name
        return s

    def list(self):
        c = []
        for cat in self.confrontation.keys():
            for area in self.confrontation[cat].keys():
                for obs in self.confrontation[cat][area]:
                    c.append(obs)

        return c

if __name__ == "__main__":
    C = Confrontation()
    print C
    for c in C.list():
        print c.name
