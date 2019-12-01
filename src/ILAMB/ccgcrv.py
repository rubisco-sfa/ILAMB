#!/usr/bin/python
"""
Program for applying curve fitting/filtering of time series data and
printing results.

Takes an input file containing two columns of data, a decimal date and value.
Applies curve fitting algorithm to the data, and depending on options,
prints results to stdout (default) or to files if specified.

General python requirements:
	numpy, scipy, dateutil

ccgg python requirements:
	ccgfilt - curve fitting/filtering
	ccgdates - date conversion routines
"""

from __future__ import print_function

import sys
import optparse

from dateutil.parser import parse
from dateutil.rrule import rrule, DAILY

import ccgfilt
from ccgdates import calendarDate, decimalDateFromDatetime, datetimeFromDecimalDate

##########################################################################
def check_export(options):
	""" Check if any options are set for filtered curve data. """

	a = options.orig \
               or options.func \
               or options.poly \
               or options.smooth \
               or options.trend \
               or options.detrend \
               or options.smcycle \
               or options.harm \
               or options.res \
               or options.smres \
               or options.trres \
               or options.ressm \
               or options.gr

	return a


##########################################################################
def export_data(options, filt):
	""" print out the curve data """

	if options.sample:
		if options.samplefile:
			try:
				fp = open(options.samplefile, "w")
			except IOError as e:
				sys.exit("Can't open file for writing. %s" % e)

		else:
			fp = sys.stdout

		export_dates(options, fp, filt, filt.xp)

	if options.equal or options.user:
		if options.file:
			try:
				fp = open(options.file, "w")
			except IOError as e:
				sys.exit("Can't open file for writing. %s" % e)

		else:
			fp = sys.stdout

		if options.equal:
			# Create a new list of dates at sample interval to give to export_dates().
			# Not quite the same as filt.xinterp because it takes into account leap years
			dates = rrule(DAILY, interval=int(filt.sampleinterval), dtstart=options.startdate, until=options.lastdate)
			xdates = [decimalDateFromDatetime(dt) for dt in dates]
			if xdates[-1] > filt.xp[-1]:
				xdates[-1] = filt.xp[-1]  # avoid problems with rounding and interpolation in ccgfilt
			if xdates[0] < filt.xp[0]:
				xdates[0] = filt.xp[0]  # avoid problems with rounding and interpolation in ccgfilt

		else:
			f = open(options.user)
			xdates = [float(line.split()[0]) for line in f]
			f.close()


		export_dates(options, fp, filt, xdates)


##########################################################################
def export_dates(options, fp, filt, x):
	""" Export (print) data to file pointer fp at dates given by x.
	The values to print are given as boolean flags in the export class.
	Some values can only be printed at sample dates, i.e. original data and residuals
	"""

	if options.showheader:
		export_header(options, fp)

	frmt = "%13.6e"

	h = filt.getHarmonicValue(x)	# harmonics
	p = filt.getPolyValue(x)	# poly
	s = filt.getSmoothValue(x)	# function + short term smoothing
	t = filt.getTrendValue(x)	# poly + long term smoothing
	g = filt.getGrowthRateValue(x)	# growth rate, derivative of trend
	f = filt.getFunctionValue(x)    # function, poly + harmonics

	for i, xp in enumerate(x):
		if options.cal:
			(yr, mon, dy, hr, mn, sec) = calendarDate(xp)
			if options.hour:
				print("%4d %02d %02d %2d" % (yr, mon, dy, hr), end='', file=fp)
			else:
				print("%4d %02d %02d" % (yr, mon, dy), end='', file=fp)
		else:
			print("%13.8f" % xp, end='', file=fp)

		if options.sample and options.orig:    print(frmt % filt.yp[i], end='', file=fp)
		if options.func:                       print(frmt % f[i], end='', file=fp)
		if options.poly:                       print(frmt % p[i], end='', file=fp)
		if options.smooth:                     print(frmt % s[i], end='', file=fp)
		if options.trend:                      print(frmt % (t[i]), end='', file=fp)
		if options.sample and options.detrend: print(frmt % (filt.yp[i] - t[i]), end='', file=fp)
		if options.smcycle:                    print(frmt % (s[i] - t[i]), end='', file=fp)
		if options.harm:                       print(frmt % (h[i]), end='', file=fp)
		if options.sample and options.res:     print(frmt % (filt.yp[i] - f[i]), end='', file=fp)
		if options.smres:                      print(frmt % (s[i] - f[i]), end='', file=fp)
		if options.trres:                      print(frmt % (t[i] - p[i]), end='', file=fp)
		if options.sample and options.ressm:   print(frmt % (filt.yp[i] - s[i]), end='', file=fp)
		if options.gr:                         print(frmt % (g[i]), end='', file=fp)

		print(file=fp)


##########################################################################
def export_header(options, fp):
	""" Export a line with column header names to file pointer fp.
	"""

	frmt = "%-13s"

	print(frmt % "date", end='', file=fp)

	# make sure these are in same order as in export_dates()
	if options.sample and options.orig:    print(frmt % "value", end='', file=fp)
	if options.func:                       print(frmt % "function", end='', file=fp)
	if options.poly:                       print(frmt % "polynomial", end='', file=fp)
	if options.smooth:                     print(frmt % "smooth", end='', file=fp)
	if options.trend:                      print(frmt % "trend", end='', file=fp)
	if options.sample and options.detrend: print(frmt % "detrended", end='', file=fp)
	if options.smcycle:                    print(frmt % "smooth_cycle", end='', file=fp)
	if options.harm:                       print(frmt % "harmonics", end='', file=fp)
	if options.sample and options.res:     print(frmt % "residuals", end='', file=fp)
	if options.smres:                      print(frmt % "smooth_resid", end='', file=fp)
	if options.trres:                      print(frmt % "trend_resid", end='', file=fp)
	if options.sample and options.ressm:   print(frmt % "resid_smooth", end='', file=fp)
	if options.gr:                         print(frmt % "growth_rate", end='', file=fp)

	print(file=fp)

#########################################################################
def read_data(filename=None):
	"""
	# Read in the input data file.
	# Format is always two columns,
	# the first column a decimal date value, (e.g. 2010.5 is halfway through 2010)
	# the second column is the corrsponding measurement value.
	"""

	if filename is None:
		fp = sys.stdin
	else:
		try:
			fp = open(filename)
		except IOError as e:
			sys.exit("Cannot open input file. %s" % e)


	x = []
	y = []
	for line in fp:
		(xv, yv) = line.split()
		x.append(float(xv))
		y.append(float(yv))

	fp.close()


	return x, y


#########################################################################

startdate = None

parser = optparse.OptionParser(usage='%prog [options] [filename]', description="Apply curve fitting/filtering to input data and print results. ")

group = optparse.OptionGroup(parser, "Filter Options")
group.add_option('--npoly', default=3, type='int', help="Number of polynomial terms in function.")
group.add_option('--nharm', default=4, type='int', help="Number of harmonic terms in function.")
group.add_option('--interv', default=0, type='float', help="Sampling interval of the data in days.")
group.add_option('--short', default=80, type='int', help="Short-term filter cutoff in days.")
group.add_option('--long', default=667, type='int', help="Long-term filter cutoff in days.")
group.add_option('--gap', default=0, type='int', help="Fill gaps larger than GAP days with function value instead of linear interpolation.")
group.add_option('--gain', action="store_true", default=False, help="Use seasonal amplitude gain factor in function fit.")
group.add_option('--timez', type='float', help="Specify time zero for coefficients of function. Default is year of first data point.")
parser.add_option_group(group)

group = optparse.OptionGroup(parser, "Output Options")
group.add_option('-f', '--file', help="Write equally spaced or user spaced output data to file instead of stdout.")
group.add_option('-s', '--samplefile', help="Write sample spaced output data to file instead of stdout.")
group.add_option('--equal', action="store_true", default=False, help="Output data at equal intervals.")
group.add_option('--sample', action="store_true", default=False, help="Output data at sample data times (default).")
group.add_option('--cal', action="store_true", default=False, help="Output dates in calendar format.")
group.add_option('--hour', action="store_true", default=False, help="Include hour in calendar format.")
group.add_option('--date', help="Output data starting at date.")
group.add_option('--user', help="Output data based on user supplied dates in file.")
group.add_option('--showheader', action="store_true", default=False, help="Include header on output to identify columns.")
parser.add_option_group(group)


group = optparse.OptionGroup(parser, "Output Parameters to include")
group.add_option('--orig', action="store_true", default=False, help="Output original data points.")
group.add_option('--func', action="store_true", default=False, help="Output function values.")
group.add_option('--poly', action="store_true", default=False, help="Output polynomial values.")
group.add_option('--smooth', action="store_true", default=False, help="Output smoothed data.")
group.add_option('--trend', action="store_true", default=False, help="Output long term trend values.")
group.add_option('--detrend', action="store_true", default=False, help="Output detrended values.")
group.add_option('--smcycle', action="store_true", default=False, help="Output smoothed, detrended annual cycle.")
group.add_option('--harm', action="store_true", default=False, help="Output values of annual harmonic functions.")
group.add_option('--res', action="store_true", default=False, help="Output residuals from the function.")
group.add_option('--smres', action="store_true", default=False, help="Output smoothed residuals from the function.")
group.add_option('--trres', action="store_true", default=False, help="Output long-term smoothed residuals from the function.")
group.add_option('--ressm', action="store_true", default=False, help="Output residuals from the smoothed curve.")
group.add_option('--gr', action="store_true", default=False, help="Output growth rate values.")
group.add_option('--coef', help="Output coefficients from index num1 to index num2. e.g. 1,4")
group.add_option('--stats', action="store_true", default=False, help="Output table of summary statistics for curve fit.")
group.add_option('--amp', action="store_true", default=False, help="Output table of statistics for annual amplitudes.")
group.add_option('--mm', action="store_true", default=False, help="Output Monthly means computed using smooth curve.")
group.add_option('--annual', action="store_true", default=False, help="Output Annual means computed using smooth curve.")
parser.add_option_group(group)

options, args = parser.parse_args()



if options.npoly < 0 or options.npoly > 10:
	sys.exit("Error in --npoly argument: value out of range (0-10) %s" % options.npoly)

if options.nharm < 0 or options.nharm > 10:
	sys.exit("Error in --npoly argument: value out of range (0-10) %s" % options.nharm)

if options.interv < 0:
	sys.exit("Error in --interval argument: value out of range (>=0) %s" % options.interv)

if options.short < 0:
	sys.exit("Error in --short argument: value out of range (must be >=0) %s" % options.short)

if options.long < 0:
	sys.exit("Error in --long argument: value out of range ( must be >=0) %s" % options.long)

if options.gap < 0:
	sys.exit("Error in --gap argument: value out of range (must be >=0) %s" % options.gap)

if options.date:
	try:
		startdate = parse(options.date)
	except ValueError:
		sys.exit("Can not get valid date from --date argument '%s'" % options.date)



if options.coef:
	try:
		(begcoef, endcoef) = options.coef.split(",")
		begcoef = int(begcoef)
		endcoef = int(endcoef)
	except ValueError:
		sys.exit("Cannot get coefficient range.")



if not len(args):
	xp, yp = read_data()
else:
	inputfile = args[0]
	xp, yp = read_data(inputfile)


# if user dates or equal spaced dates aren't specified, use sample dates as default
if not options.user and not options.equal: options.sample = True

# Compute the filtered data
if options.timez is None: options.timez = int(xp[0])
filt = ccgfilt.ccgFilter(xp, yp, options.short, options.long, options.interv, options.npoly, options.nharm, options.timez, options.gap, options.gain)


# If starting date is not specified, set it to the date of the first data point
# Set ending date to date of last data point
if startdate is None:
	options.startdate = datetimeFromDecimalDate(filt.xp[0])
else:
	options.startdate = startdate
options.lastdate = datetimeFromDecimalDate(filt.xp[-1])

if check_export(options):
	export_data(options, filt)

if options.amp:
	months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
	amps = filt.getAmplitudes()

	print(" *****  Seasonal Cycle Statistics.  *****")
	print(" Year      Amplitude     Maximum   Date     Minimum   Date")
	print("-----------------------------------------------------------")

	frmt = "%5.0f %12.2f %12.2f   %3s %2d %9.2f   %3s %2d"
	for (year, amp, maxdate, maxval, mindate, minval) in amps:

		(yr, mnmax, dmax, hr, mn, sec) = calendarDate(maxdate)
		(yr, mnmin, dmin, hr, mn, sec) = calendarDate(mindate)

		print(frmt % (year, amp, maxval, months[mnmax], dmax, minval, months[mnmin], dmin))

if options.stats:
	print(filt.stats())

if options.mm:
	mm = filt.getMonthlyMeans()
	for (year, month, val, std, n) in mm:
		print("%4d %02d %7.2f %5.2f %2d" % (year, month, val, std, n))

if options.annual:
	am = filt.getAnnualMeans()
	for (year, val, std, n) in am:
		print("%4d %7.2f %5.2f %2d" % (year, val, std, n))

if options.coef:
	for i in range(begcoef, min(filt.numpm, endcoef)):
		print("%.6f" % filt.params[i], end='')
	print()
