Package Contents
================

We have written this python package with the intent that it be easy to use on a variety of levels. For the researcher looking to do his/her own analysis, we provide a low-level library to make interfacing/manipulating with model/observational data simple and efficient. We also provide an abstract model result object for simplifying model queries and organizing results when confronting models with observational data. Finally we encapsulate the observational data and model comparison routines into an abstract object called a confrontation. This design aims to make extending the package simpler, not requiring the user to understand the entire code.

ilamblib
--------

This first layer is a module which provides python implementations of common operations performed in operating and analyzing data. They are organized here by categories to help give a sense for how they are intended to be used.

Miscellaneous
~~~~~~~~~~~~~

.. currentmodule:: ILAMB.ilamblib
.. autosummary::
   :toctree: _generated

   ExtractPointTimeSeries
   ExtractTimeSeries
   GenerateDistinctColors
   MonthlyWeights
   CellAreas
   GlobalLatLonGrid

Metrics
~~~~~~~

.. autosummary::
   :toctree: _generated

   Bias
   RootMeanSquaredError
   NearestNeighborInterpolation
   TrueError

Analysis
~~~~~~~~

.. autosummary::
   :toctree: _generated

   AnnualMean
   AnnualMinMax
   AnnualCycleInformation
   DecadalAmplitude
   DecadalMaxTime
   DecadalMinTime
   WindowedTrend
   SpatiallyIntegratedTimeSeries
   TemporallyIntegratedTimeSeries

Variable
~~~~~~~~

.. currentmodule:: ILAMB.Variable
.. autosummary::
   :toctree: _generated
   :nosignatures:
   :template: class.rst

   Variable

ModelResults
------------

This model result class aims to handle model queries efficiently, and uniformly without requiring the user to interface with the source netCFD files themselves. The object will also contain confrontation outputs which can then be post-processed in a variety of ways. For details on the ModelResult object, click on its link below.

.. currentmodule:: ILAMB.ModelResult
.. autosummary::
   :toctree: _generated
   :nosignatures:
   :template: class.rst
   
   ModelResult

Confrontations
--------------

The idea of a confrontation class is that we would like to pair the observational data with the routines used to confront models with that data. In this way, the confrontations are self-contained and modular. As with ModelResults, each of these links in a separate class and has its own documentation which you can explore by following the link.

.. currentmodule:: ILAMB.Confrontation
.. autosummary::
   :toctree: _generated
   :nosignatures:
   :template: class.rst
   
   Confrontation

.. currentmodule:: ILAMB.CO2MaunaLoa
.. autosummary::
   :toctree: _generated
   :nosignatures:
   :template: class.rst

   CO2MaunaLoa

.. currentmodule:: ILAMB.GPPFluxnetGlobalMTE
.. autosummary::
   :toctree: _generated
   :nosignatures:
   :template: class.rst

   GPPFluxnetGlobalMTE

Post
----

Finally we provide a module with many ways of presenting these results. While we want to keep data formats open such that researchers may use plotting code of their own in their own language, we also want to provide a robust set of tools for generating different views and organizations of the confrontation results. 

.. currentmodule:: ILAMB.Post
.. autosummary::
   :toctree: _generated

   ConfrontationTableASCII
   ConfrontationTableGoogle
   GlobalPlot
   ColorBar
