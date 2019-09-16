from .Confrontation import Confrontation
from .Variable import Variable
from .Regions import Regions
from . import Post as post
from . import ilamblib as il
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def AnalysisUncertaintySpatial(ref,com,**keywords):
    """blah
    """
    regions            = keywords.get("regions"           ,["global"])
    dataset            = keywords.get("dataset"           ,None)
    benchmark_dataset  = keywords.get("benchmark_dataset" ,None)
    space_mean         = keywords.get("space_mean"        ,True)
    table_unit         = keywords.get("table_unit"        ,None)
    plot_unit          = keywords.get("plot_unit"         ,None)
    mass_weighting     = keywords.get("mass_weighting"    ,False)
    expert_uncertainty = keywords.get("expert_uncertainty",None)            
    use_expert         = keywords.get("use_expert"        ,False)
    ILAMBregions       = Regions()
    spatial            = ref.spatial
    name = ref.name
    ref_timeint = None
    com_timeint = None
                
    # Interpolate both reference and comparison to a grid composed of
    # their cell breaks
    ref.convert(plot_unit)
    com.convert(plot_unit)
    lat,lon,lat_bnds,lon_bnds = il._composeGrids(ref,com)
    REF   = ref.interpolate(lat=lat,lon=lon,lat_bnds=lat_bnds,lon_bnds=lon_bnds)
    COM   = com.interpolate(lat=lat,lon=lon,lat_bnds=lat_bnds,lon_bnds=lon_bnds)
    unit  = REF.unit
    area  = REF.area
    ndata = REF.ndata

    # Find the mean values over the time period
    if ref_timeint is None:
        ref_timeint = ref.integrateInTime(mean=True).convert(plot_unit)
        REF_timeint = REF.integrateInTime(mean=True).convert(plot_unit)
    else:
        ref_timeint.convert(plot_unit)
        REF_timeint = ref_timeint.interpolate(lat=lat,lon=lon,lat_bnds=lat_bnds,lon_bnds=lon_bnds)
    if com_timeint is None:
        com_timeint = com.integrateInTime(mean=True).convert(plot_unit)
        COM_timeint = COM.integrateInTime(mean=True).convert(plot_unit)
    else:
        com_timeint.convert(plot_unit)
        COM_timeint = com_timeint.interpolate(lat=lat,lon=lon,lat_bnds=lat_bnds,lon_bnds=lon_bnds)
    normalizer  = REF_timeint.data if mass_weighting else None

    # Report period mean values over all possible representations of
    # land
    ref_and_com = (REF_timeint.data.mask == False) * (COM_timeint.data.mask == False)
    ref_not_com = (REF_timeint.data.mask == False) * (COM_timeint.data.mask == True )
    com_not_ref = (REF_timeint.data.mask == True ) * (COM_timeint.data.mask == False)
    if benchmark_dataset is not None:

        ref_timeint.name = "timeint_of_%s" % name
        ref_timeint.toNetCDF4(benchmark_dataset,group="MeanState")
        for region in regions:

            # reference period mean on original grid
            ref_period_mean = ref_timeint.integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            ref_period_mean.name = "Period Mean (original grids) %s" % region
            ref_period_mean.toNetCDF4(benchmark_dataset,group="MeanState")

    if dataset is not None:

        com_timeint.name = "timeint_of_%s" % name
        com_timeint.toNetCDF4(dataset,group="MeanState")
        for region in regions:

            # reference period mean on intersection of land
            ref_union_mean = Variable(name = "REF_and_com", unit = REF_timeint.unit,
                                      data = np.ma.masked_array(REF_timeint.data,mask=(ref_and_com==False)),
                                      lat  = lat, lat_bnds = lat_bnds, lon  = lon, lon_bnds = lon_bnds,
                                      area = REF_timeint.area).integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            ref_union_mean.name = "Benchmark Period Mean (intersection) %s" % region
            ref_union_mean.toNetCDF4(dataset,group="MeanState")

            # reference period mean on complement of land
            ref_comp_mean = Variable(name = "REF_not_com", unit = REF_timeint.unit,
                                     data = np.ma.masked_array(REF_timeint.data,mask=(ref_not_com==False)),
                                     lat  = lat, lat_bnds = lat_bnds, lon  = lon, lon_bnds = lon_bnds,
                                     area = REF_timeint.area).integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            if ref_comp_mean.data.mask: ref_comp_mean.data = np.ma.masked_array([0.])
            ref_comp_mean.name = "Benchmark Period Mean (complement) %s" % region
            ref_comp_mean.toNetCDF4(dataset,group="MeanState")

            # comparison period mean on original grid
            com_period_mean = com_timeint.integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            com_period_mean.name = "Period Mean (original grids) %s" % region
            com_period_mean.toNetCDF4(dataset,group="MeanState")

            # comparison period mean on intersection of land
            com_union_mean = Variable(name = "ref_and_COM", unit = COM_timeint.unit,
                                      data = np.ma.masked_array(COM_timeint.data,mask=(ref_and_com==False)),
                                      lat  = lat, lat_bnds = lat_bnds, lon  = lon, lon_bnds = lon_bnds,
                                      area = COM_timeint.area).integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            com_union_mean.name = "Model Period Mean (intersection) %s" % region
            com_union_mean.toNetCDF4(dataset,group="MeanState")

            # comparison period mean on complement of land
            com_comp_mean = Variable(name = "COM_not_ref", unit = COM_timeint.unit,
                                     data = np.ma.masked_array(COM_timeint.data,mask=(com_not_ref==False)),
                                     lat  = lat, lat_bnds = lat_bnds, lon  = lon, lon_bnds = lon_bnds,
                                     area = COM_timeint.area).integrateInSpace(region=region,mean=space_mean).convert(table_unit)
            if com_comp_mean.data.mask: com_comp_mean.data = np.ma.masked_array([0.])
            com_comp_mean.name = "Model Period Mean (complement) %s" % region
            com_comp_mean.toNetCDF4(dataset,group="MeanState")

    # Now that we are done reporting on the intersection / complement,
    # set all masks to the intersection
    REF.data.mask += np.ones(REF.time.size,dtype=bool)[:,np.newaxis,np.newaxis] * (ref_and_com==False)
    COM.data.mask += np.ones(COM.time.size,dtype=bool)[:,np.newaxis,np.newaxis] * (ref_and_com==False)
    REF_timeint.data.mask = (ref_and_com==False)
    COM_timeint.data.mask = (ref_and_com==False)
    if mass_weighting: normalizer.mask = (ref_and_com==False)

    # Bias: maps, scalars, and scores
    bias = REF_timeint.bias(COM_timeint).convert(plot_unit)
    cREF = Variable(name = "centralized %s" % name, unit = REF.unit,
                    data = np.ma.masked_array(REF.data-REF_timeint.data[np.newaxis,...],mask=REF.data.mask),
                    time = REF.time, time_bnds = REF.time_bnds, ndata = REF.ndata,
                    lat  = lat, lat_bnds = lat_bnds, lon = lon, lon_bnds = lon_bnds, area = REF.area).convert(plot_unit)
    REF_std = cREF.rms()
    bias_score_map = il.Score(bias,REF_std if REF.time.size > 1 else REF_timeint)
    bias_score_map.data.mask = (ref_and_com==False) # for some reason I need to explicitly force the mask
    if dataset is not None:
        bias.name = "bias_map_of_%s" % name
        bias.toNetCDF4(dataset,group="MeanState")
        bias_score_map.name = "biasscore_map_of_%s" % name
        bias_score_map.toNetCDF4(dataset,group="MeanState")
        for region in regions:
            bias_val = bias.integrateInSpace(region=region,mean=True).convert(plot_unit)
            bias_val.name = "Bias %s" % region
            bias_val.toNetCDF4(dataset,group="MeanState")
            bias_score = bias_score_map.integrateInSpace(region=region,mean=True,weight=normalizer)
            bias_score.name = "Bias Score %s" % region
            bias_score.toNetCDF4(dataset,group="MeanState")
    del bias,bias_score,cREF
    
    # Changes with uncertainty ----------------------------------------------------------------
    
    # convert units of the expert value is present
    if expert_uncertainty is not None:
        match = re.search("(-?\d*\.?\d*)\s(.*)",expert_uncertainty.strip())
        if match:
            expert_uncertainty = Variable(data = np.asarray(float(match.group(1))),
                                          unit = match.group(2)).convert(ref.unit).data
        else:
            msg = "The option 'expert_uncertainty' must follow a VALUE UNIT format, where the UNIT must be convertible to the reference dataset unit"
            raise ValueError(msg)
            
    # perform some logic/checks on what uncertainty to use
    if use_expert:
        if expert_uncertainty is None:
            msg = "The 'use_expert' option was given, but no 'expert_uncertainty' specified in the configure file"
            raise ValueError(msg)
        Dref = expert_uncertainty
    else:
        if REF.data_bnds is None:
            if expert_uncertainty is None:
                msg = "No uncertainty is specified for this variable, try specifying 'expert_uncertainty' in the configure file"
                raise ValueError(msg)
            Dref = expert_uncertainty
        else:
            Dref = np.abs(REF.data_bnds[...,0]-REF.data)

    with np.errstate(under='ignore',over='ignore'):
        bias_uscore_map = np.exp(-(np.abs( REF.data-COM.data)-Dref).clip(0) / REF_std.data[np.newaxis,...])
    bias_uscore_map = Variable(name = "biasuscore_map_of_u%s" % name,
                               unit = "1",
                               data = bias_uscore_map,
                               time = REF.time, time_bnds = REF.time_bnds,
                               lat  = REF.lat, lat_bnds = REF.lat_bnds,
                               lon  = REF.lon, lon_bnds = REF.lon_bnds,
                               area = REF.area, ndata = REF.ndata).integrateInTime(mean=True)
    if benchmark_dataset is not None:
        if Dref.size == 1:
            Variable(name = "Expert Uncertainty",
                     unit = REF.unit,
                     data = Dref).toNetCDF4(benchmark_dataset,group="MeanState")
        else:
            Dref = Variable(name = "uncertain",
                            unit = ref.unit,
                            data = Dref,
                            time = REF.time, time_bnds = REF.time_bnds,
                            lat  = REF.lat, lat_bnds = REF.lat_bnds,
                            lon  = REF.lon, lon_bnds = REF.lon_bnds,
                            area = REF.area, ndata = REF.ndata).integrateInTime(mean=True)
            Dref.name = "uncertain"
            Dref.toNetCDF4(benchmark_dataset,group="MeanState")

    if dataset is not None:
        bias_uscore_map.name = "biasuscore_map_of_%s" % name
        bias_uscore_map.toNetCDF4(dataset,group="MeanState")
        for region in regions:
            bias_uscore = bias_uscore_map.integrateInSpace(region=region,mean=True,weight=normalizer)
            bias_uscore.name = "Uncertainty Bias Score %s" % region
            bias_uscore.toNetCDF4(dataset,group="MeanState")
            
class ConfUncertainty(Confrontation):

    def confront(self,m):

        # parse options
        expert_uncertainty = self.keywords.get("expert_uncertainty",None)            
        use_expert         = self.keywords.get("use_expert"        ,"False").lower() == "true"
        
        # get reference and model data
        ref,mod = self.stageData(m)

        mod_file = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        ref_file = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name,      ))
        with il.FileContextManager(self.master,mod_file,ref_file) as fcm:

            # Encode some names and colors
            fcm.mod_dset.setncatts({"name" :m.name,
                                    "color":m.color,
                                    "complete":0})
            if self.master:
                fcm.obs_dset.setncatts({"name" :"Benchmark",
                                        "color":np.asarray([0.5,0.5,0.5]),
                                        "complete":0})
            AnalysisUncertaintySpatial(ref,mod,dataset    = fcm.mod_dset,
                                       regions            = self.regions,
                                       benchmark_dataset  = fcm.obs_dset,
                                       table_unit         = self.table_unit,
                                       plot_unit          = self.plot_unit,
                                       space_mean         = self.space_mean,
                                       expert_uncertainty = expert_uncertainty,
                                       use_expert         = use_expert)
            fcm.mod_dset.setncattr("complete",1)
            if self.master: fcm.obs_dset.setncattr("complete",1)
        
    def modelPlots(self,m):
        super(ConfUncertainty,self).modelPlots(m)
        for page in self.layout.pages:
            for sec in page.figures.keys():
                for fig in page.figures[sec]:
                    fig.side = fig.side.replace("BIAS SCORE","ORIGINAL BIAS SCORE")
        bname = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name       ))
        fname = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return

        v = None
        try:
            v = Variable(filename = bname, variable_name = "uncertain", groupname = "MeanState")
        except:
            pass
        if v is not None:
            pname = "uncertain"
            page.addFigure("Temporally integrated period mean",
                           pname,
                           "MNAME_RNAME_uncertain.png",
                           side = "BENCHMARK UNCERTAINTY",
                           legend = True)
            vmin,vmax = np.percentile(v.data.compressed(),[1,99])
            
            fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
            post.ColorBar(ax,
                          vmin = vmin,
                          vmax = vmax,
                          cmap = "Reds",
                          label = v.unit)
            fig.savefig(os.path.join(self.output_path,"legend_%s.png" % (pname)))
            plt.close()
            
            for region in self.regions:
                ax = v.plot(None,
                            region = region,
                            vmin   = vmin,
                            vmax   = vmax,
                            cmap   = "Reds")
                fig = ax.get_figure()
                fig.savefig(os.path.join(self.output_path,"%s_%s_%s.png" % (m.name,region,pname)))
                plt.close()

        v = None
        try:
            v = Variable(filename = fname, variable_name = "biasuscore_map_of_%s" % self.variable, groupname = "MeanState")
        except:
            pass
        if v is not None:
            pname = "biasuscore"
            page.addFigure("Temporally integrated period mean",
                           pname,
                           "MNAME_RNAME_%s.png" % pname,
                           side = "UNCERTAINTY BIAS SCORE",
                           legend = True)            
            fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
            post.ColorBar(ax,
                          vmin = 0,
                          vmax = 1,
                          cmap = "RdYlGn",
                          label = "1")
            fig.savefig(os.path.join(self.output_path,"legend_%s.png" % (pname)))
            plt.close()
            
            for region in self.regions:
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                v.plot(ax,
                       region = region,
                       vmin   = 0,
                       vmax   = 1,
                       cmap   = "RdYlGn")
                fig.savefig(os.path.join(self.output_path,"%s_%s_%s.png" % (m.name,region,pname)))
                plt.close()
