from numpy import asarray,ones,copy as npcopy

__all__ = ['spm','mph','hpd','mpy','dpy_noleap','dpy_gregorian','dpy_360','dpm_noleap','dpm_gregorian','dpm_360','g_per_Pg','g_per_kg','Ar_molar_mass','C_molar_mass','N_molar_mass','O_molar_mass','CO2_molar_mass','dry_air_molar_mass','dry_air_mass','dry_air_moles','co2_g_per_ppm','co2_ppm_per_kg','co2_ppm_per_C_Pg']

# Time constants
spm              = 60.     # seconds per minute
mph              = 60.     # minutes per hour
hpd              = 24.     # hours per day
mpy              = 12.     # months per year
dpy_noleap       = 365.0   # days per year (for no leap year calendars)
dpy_gregorian    = 365.25  # days per year
dpy_360          = 360.0   # days per year (for 30 days/month)
dpm_noleap       = asarray([31,28,31,30,31,30,31,31,30,31,30,31],dtype='float') # days per month
dpm_gregorian    = npcopy(dpm_noleap) ; dpm_gregorian[1] = dpm_gregorian[1] + 0.25
dpm_360          = ones(mpy)*30.

# Mass unit conversions
g_per_Pg         = 1e+15   # grams per Pg
g_per_kg         = 1e+3    # grams per kg

# Chemical constants
Ar_molar_mass    = 39.948  # grams per mole
C_molar_mass     = 12.0107 # grams per mole
N_molar_mass     = 14.0067 # grams per mole
O_molar_mass     = 15.9994 # grams per mole
CO2_molar_mass   = C_molar_mass + 2. * O_molar_mass # grams per mole

# Atmospheric constants
dry_air_molar_mass = 0.78084*2.*N_molar_mass + 0.20946*2.*O_molar_mass + 0.00934*Ar_molar_mass + 0.00039445*CO2_molar_mass # grams per mole
dry_air_mass       = 5.1352e+21 # grams
dry_air_moles      = dry_air_mass / dry_air_molar_mass
co2_g_per_ppm      = dry_air_moles * CO2_molar_mass / 1.e+6
co2_ppm_per_kg     = g_per_kg / co2_g_per_ppm
co2_ppm_per_C_Pg   = g_per_Pg / co2_g_per_ppm * CO2_molar_mass/C_molar_mass


convert = {}
convert["co2"] = {}
convert["co2"]["1e-6"] = {"kg"  :co2_ppm_per_kg, "1e-6": 1.}
convert["co2"]["kg"]   = {"1e-6":1./co2_ppm_per_kg, "kg": 1.}
