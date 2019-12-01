
import datetime
import numpy

###########################################################
def datesOk(year, month, day, hour=0, minute=0, second=0):
	""" Check if values are appropriate for a real date """

	if month < 1 or month > 12:
		raise ValueError("Month is out of range")

	if day < 1 or day > 31:
		raise ValueError("Day is out of range. Date values given are %d %d %d %d %d %d" % (year, month, day, hour, minute, second))

	if hour < 0 or hour > 23:
		raise ValueError("Hour is out of range")

	if minute < 0 or minute > 59:
		raise ValueError("Minute is out of range")

	if second < 0 or second > 59:
		raise ValueError("Second is out of range")

	return True

###########################################################
def decimalDate (year, month, day, hour=0, minute=0, second=0):
	""" Convert a date and time to a fractional year. """

	if not datesOk(year, month, day, hour, minute, second):
		return 0

	soy = secondOfYear(year, month, day, hour, minute, second)

	if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
		dd = year + soy/3.16224e7
	else:
		dd = year + soy/3.1536e7

	return dd

###################################################
def secondOfYear(year, month, day, hour, minute, second):

	if not datesOk(year, month, day, hour, minute, second):
		return 0

	doy = dayOfYear(year, month, day)
	soy = (doy-1)*86400 + hour*3600 + minute*60 + second

	return soy

###################################################
def dayOfYear(year, month, day):

	if not datesOk(year, month, day):
		return 0

	mona =  [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

	x = mona[month-1] + day
	if ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0) and month > 2:
		x = x + 1

	return x

###################################################
# Make an integer value that contains the date and hour
def intDate (year, month, day, hour=0):

	if not datesOk(year, month, day, hour):
		return 0

	id = year*1000000 + month*10000 + day*100 + hour

	return id

###################################################
# Opposite of intDate, get date and hour from integer value
def getDate(intdate):

	s = str(intdate)

	if len(s) != 10:
		raise ValueError("Invalid date")

	year = int(s[0:4])
	month = int(s[4:6])
	day = int(s[6:8])
	hour = int(s[8:])

	return year, month, day, hour

###################################################
# Don't use these routines for resolution less than 1 second.
def calendarDate (decyear):

	dyr =  int(decyear)
	fyr = decyear - dyr

	if dyr % 4 == 0:
		nsec = fyr * (366*86400)
	else:
		nsec  = fyr * (365*86400)

	nsec = round(nsec, 0)

	ndays = int(nsec / 86400)
	doy  = ndays + 1

	if doy > 366:
		dyr = dyr + 1
		doy = 1
	month, day = to_mmdd(dyr, doy)


	nsecs = round(nsec - (ndays*86400), 0)
	hour  = int(nsecs / 3600)
	minute = int((nsecs - (hour *3600)) / 60)
	seconds = int(round(nsecs - (hour * 3600.0) - (minute * 60.0), 0))

	return dyr,month,day,hour,minute,seconds


###################################################
def to_mmdd (year, doy):

	if doy < 1 or doy > 366:
		raise ValueError("Day of year is out of range")

	if year % 4 == 0:
		mona = [31,29,31,30,31,30,31,31,30,31,30,31]
	else:
		mona  = [31,28,31,30,31,30,31,31,30,31,30,31]

	idoy =  doy
	imon = 0
	while idoy - mona[imon] > 0:
		idoy  = idoy - mona[imon]
		imon += 1

	month = imon + 1
	return int(month), int(idoy)

###################################################
def toMonthDay(year, doy):

	if doy < 1 or doy > 366:
		raise ValueError("Day of year is out of range")

	month, day = to_mmdd(year, doy)

	return month, day

###################################################
def getDatetime(datestr, sep=""):
	""" create a datetime from yyyy mm dd string """

	separators = ["", "-", "/"]
	if sep:
		separators = separators + sep


	for separator in separators:
		if separator == "":
			a = datestr.split()
		else:
			a = datestr.split(separator)

		if len(a) >= 3:
			year = int(a[0])
			month = int(a[1])
			day = int(a[2])
			if len(a)>3:
				hour = int(a[3])
				minute = int(a[4])
				if len(a) == 6:
					second = int(a[5])
				else:
					second = 0
			else:
				hour = 0
				minute = 0
				second = 0

			try:
				ok = datesOk(year, month, day, hour, minute)
			except ValueError as msg:
				raise ValueError("Cannot create datetime from string: %s %s" % (datestr, msg))

			date = datetime.datetime(year, month, day, hour, minute, second)
			return date


	raise ValueError("Cannot create datetime from string: %s" % datestr)

	return None

###################################################
def getTime(timestr, sep=""):

	separators = ["", ":"]
	if sep:
		separators = separators + sep


	for separator in separators:
		if separator == "":
			a = timestr.split()
		else:
			a = timestr.split(separator)

		if len(a) >= 3:

			hour = int(a[0])
			minute = int(a[1])
			second = int(float(a[2]))

			time = datetime.time(hour, minute, second)
			return time

	raise ValueError("Cannot create time from string: %s" % timestr)

	return None
			

###################################################
def decimalDateFromDatetime(dt):

	dd = decimalDate(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

	return dd

###################################################
def datetimeFromDateAndTime(d, t):

	s = "%s" % t
	(hr,mn,sc) = s.split(":")
	
	dt = datetime.datetime(d.year, d.month, d.day, int(hr), int(mn), int(sc))

	return dt

###################################################
def datetimeFromDecimalDate(dd):

	yr, mon, dy, hr, mn, sc = calendarDate(dd)
	dt = datetime.datetime(yr, mon, dy, hr, mn, sc)

	return dt


###################################################
def dateFromDecimalDate(dd):

	yr, mon, dy, hr, mn, sc = calendarDate(dd)
	dt = datetime.date(yr, mon, dy)

	return dt

def dec2date(dd):

	a = numpy.empty((dd.size, 6))
	# dd is a numpy array of decimal dates
	for i in range(dd.size):
		yr, mon, dy, hr, mn, sc = calendarDate(dd[i])
		a[i] = (yr, mon, dy, hr, mn, sc)

	return a
