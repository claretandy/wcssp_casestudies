import categorical
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Path, PathPatch


'''
#######################################################################################################################
#######################################################################################################################
The graph.py module was created for generating basemap images and plot (i.e. line graph and bar graph) from the provided data.
These data can be a model output, observed data or a processed data (i.e. bias, rmse, categorical results, etc) which
statisfy the correct format. The user/programmer still have a freedom to edit styles (i.e. font size, alpha, labels and etc)
of the figure which creates flexibility to the module. In this class, there are two major functions: create_basemap and create_plot. 

There are five types of available model_type in this class:
- UM (Unified Model Global)
- GSM (Global Spectral Model)
- WRF_12km (WRF model 12 km)
- WRF_3km (WRF model 3 km)
- GSMaP (Global Satellite Mapping of Precipitation)

Take note, there is a possibility that an error will occur if outside of the choices in the model_type was used.
#######################################################################################################################
#######################################################################################################################

'''
class Graph:

	# Initilialize attributes of the class Graph
	def __init__(self, model_type, lat=None, lon=None, basemap=True):

		# only applicable for function create_basemap()
		if basemap:
			self.lon = lon
			self.lat = lat
			self.resolution = 'c'
			self.linewidth = 1
			self.sub_title_font_size = 10

		# attributes for create_plot function
		else:
			self.x_name = np.array(["Insert X Name"])
			self.x_label = "Insert X Label"
			self.y_label = "Insert Y Label"
			self.x_legend_name = np.array(["Insert Legend Name"])
			self.alpha = 1

		self.model_type = model_type
		self.main_title = "Insert Main Title"
		self.main_title_font_size = 12
		self.dpi = 300
		self.main_title_y = 1

	# creates a figure with a single subplot or multiple subplot depending on the graph_type. For "multiple", the maximum input data is 8 and 
	# it divides the subplots automatically
	def create_figure(self, output, graph_type):
		if type(output) == list:
			loop_index = len(output)
		elif type(output) == np.ndarray:
			loop_index = output.shape[0]

		if graph_type == "single":
			loop_index = 1
			fig, ax = plt.subplots(1,1, figsize=(8,8))
		elif graph_type == "multiple":
			if loop_index <= 2:
				fig, ax = plt.subplots(1,2,figsize=(10,8))
			elif loop_index <= 4:
				fig, ax = plt.subplots(2,2,figsize=(8,8))			
			elif loop_index <= 6:
				fig, ax = plt.subplots(2,3,figsize=(10,8))
			elif loop_index <= 8:
				fig, ax = plt.subplots(2,4,figsize=(10,6.5))

			if 'WRF_3km' in self.model_type:
				plt.subplots_adjust(top=0.85)
			elif 'UM' in self.model_type:
				plt.subplots_adjust(top=0.87)
			else:
				plt.subplots_adjust(hspace=0)

			ax = ax.ravel()

			if (loop_index % 2) !=  0:
				fig.delaxes(ax[loop_index])

		return fig, ax, loop_index

	# filter the data according to its input map boundary and get the shape of the filtered data
	def get_shape_bound(self, map_boundary, lat, lon):
		bound_lon_index = np.where(np.logical_and(lon >= map_boundary[2],lon <= map_boundary[3]))
		bound_lat_index = np.where(np.logical_and(lat >= map_boundary[0],lat <= map_boundary[1]))
		lat_bound = lat[bound_lat_index]
		lon_bound = lon[bound_lon_index]
		shape = [len(lon_bound),len(lat_bound)]

		return shape

	# creates a color map for the color bar in the create_basemap function. 
	# There are five types of color maps: bias, rmse, 3_hourly_rainfall, 6_hourly_rainfall and default
	def create_cmap(self, color_bar):
		if color_bar == 'bias' or color_bar == 'rmse':
			if color_bar == 'bias':
				cmap = plt.cm.get_cmap(name='seismic_r')
				cmaplist = [cmap(i) for i in range(cmap.N)]
				boundaries = np.arange(-100,110,20)

			elif color_bar == 'rmse':
				cmap = plt.cm.get_cmap(name='jet')
				cmaplist = [cmap(i) for i in range(cmap.N)]
				boundaries = np.arange(0,110,20)
				
			cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
			norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

		elif color_bar in ['3_hourly_rainfall', '6_hourly_rainfall']:
			rgb = [[255,255,255], [211,211,211], [255,255,0], [255,165,0], [255,0,0], [255,105,180], [148,0,211]] 
			rgb=np.array(rgb)/255.
			cmap = ListedColormap(rgb,"")

			if color_bar == '3_hourly_rainfall':
				norm_boundary = [0, 0.1, 1, 7.5, 22.5, 45, 90, 1000]
				norm = matplotlib.colors.BoundaryNorm(norm_boundary, cmap.N, clip=True)
				boundaries = [0.05,0.55,3.75,15,33.75,67.5,545]
				
			elif color_bar == '6_hourly_rainfall':
				norm_boundary = [0, 0.1, 1, 15, 45, 90, 180, 1000]
				norm = matplotlib.colors.BoundaryNorm(norm_boundary, cmap.N, clip=True)
				boundaries = [0.05,0.55,8,30,67.5,135,590]

		elif color_bar == 'default':
			rgb = [[255,255,255],[213,213,213],[173,173,173],[255,255,0],[255,140,0],[255,0,0],[158,30,30],[255,0,214],[124,0,218]]
			rgb=np.array(rgb)/255.
			cmap = ListedColormap(rgb,"")
			norm_boundary = [0, 10, 30, 50, 100, 200, 300, 400, 500, 1000000000]
			norm = matplotlib.colors.BoundaryNorm(norm_boundary, cmap.N, clip=True)
			boundaries = [5,20,40,75,150,250,350,450,500000250]

		return cmap, norm, boundaries

	''' 
		creates basemap with a focused region (controlled by the map_boundary) from the output provided and save it to the input file_path. 
		The output can be set to be visible only on land (plot_type="land_only") and shapefile can be used for its basemap instead of the default
		used by the matplotlib. If the graph_type is set to be 'multiple', the model_type attritbute, latitude and longitude must have the same number of output 
		and must be the representation of that certain output. For example, if the output contains GSMaP, UM and WRF data, the model type, latitude and longitude
		must represent each of that output since there is a possibility that each data have different latitude and longitude. 
	'''
	def create_basemap(self, file_path, output, map_boundary, sub_title=None, color_bar="default", graph_type="single", plot_type="land_water", shapefile=True):
		directory, filename = file_path.rsplit('/', 1)

		if not os.path.exists(directory):
			os.makedirs(directory)

		fig, ax, loop_index = self.create_figure(output, graph_type)

		for i in range(loop_index):
			if graph_type == "single":
				m = Basemap(projection='merc',llcrnrlat=map_boundary[0],urcrnrlat=map_boundary[1],llcrnrlon=map_boundary[2],\
				   			urcrnrlon=map_boundary[3],lat_ts=20,resolution=self.resolution)	

				data_type = self.model_type
				shape = self.get_shape_bound(map_boundary, self.lat, self.lon)

				if data_type in ["GSMaP"]:
					xmesh_model, ymesh_model = np.meshgrid(self.lon, self.lat[::-1], sparse=False)
				else:
					xmesh_model, ymesh_model = np.meshgrid(self.lon, self.lat, sparse=False)

			elif graph_type == "multiple":
				m = Basemap(projection='merc',llcrnrlat=map_boundary[0],urcrnrlat=map_boundary[1],llcrnrlon=map_boundary[2],\
				   			urcrnrlon=map_boundary[3],lat_ts=20,ax=ax[i],resolution=self.resolution)	

				data_type = self.model_type[i]
				shape = self.get_shape_bound(map_boundary, self.lat[i], self.lon[i])

				if data_type in ["GSMaP"]:
					xmesh_model, ymesh_model = np.meshgrid(self.lon[i], self.lat[i][::-1], sparse=False)
				else:
					xmesh_model, ymesh_model = np.meshgrid(self.lon[i], self.lat[i], sparse=False)

			if shapefile:
				m.readshapefile('/home/ict/Desktop/work/shapefile/gadm36_PHL_shp/province/gadm36_PHL_1', 'PH_provinces',linewidth=self.linewidth)
				m.readshapefile('/home/ict/Desktop/work/test/verification_test/graph_shp/CHN_part', 'CHN',linewidth=self.linewidth)
				m.readshapefile('/home/ict/Desktop/work/test/verification_test/graph_shp/IDN_part', 'IDN',linewidth=self.linewidth)
				m.readshapefile('/home/ict/Desktop/work/test/verification_test/graph_shp/MYS_part', 'MYS',linewidth=self.linewidth)
				m.readshapefile('/home/ict/Desktop/work/shapefile/Taiwan/gadm36_TWN_shp/gadm36_TWN_0', 'TWN',linewidth=self.linewidth)
			else:
				m.drawcoastlines(linewidth=self.linewidth)
				m.drawstates(linewidth=self.linewidth)
				m.drawcountries(linewidth=self.linewidth)

			bound = np.where(np.logical_and(np.logical_and(ymesh_model >= map_boundary[0],ymesh_model <= map_boundary[1]),\
			np.logical_and(xmesh_model >= map_boundary[2],xmesh_model <= map_boundary[3])))				

			lon_map, lat_map = m(xmesh_model[bound[0],bound[1]].reshape(shape[1],shape[0]), ymesh_model[bound[0],bound[1]].reshape(shape[1],shape[0]))

			if plot_type == 'land_only':
				if graph_type == 'single':
					x0,x1 = ax.get_xlim()
					y0,y1 = ax.get_ylim()
				elif graph_type == 'multiple':
					x0,x1 = ax[i].get_xlim()
					y0,y1 = ax[i].get_ylim()

				map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
				polys = [p.boundary for p in m.landpolygons]
				polys = [map_edges]+polys[:]
				codes = [
				    [Path.MOVETO] + [Path.LINETO for p in p[1:]]
				    for p in polys
				]
				polys_lin = [v for p in polys for v in p]
				codes_lin = [c for cs in codes for c in cs]
				path = Path(polys_lin, codes_lin)
				patch = PathPatch(path,facecolor='white', lw=0)

				if graph_type == 'single':
					ax.add_patch(patch)
				elif graph_type == 'multiple':
					ax[i].add_patch(patch)

			cmap, norm, boundaries = self.create_cmap(color_bar)

			if graph_type == "single": 
				if data_type == "GSMaP":
					output_bound = output[bound[1],bound[0]]
				else: 	
					output_bound = output[bound[0],bound[1]] 

				plt.title(self.main_title, fontsize=12, loc='center')
				cs = ax.pcolormesh(lon_map,lat_map,output_bound.reshape(shape[1],shape[0]), cmap=cmap, norm=norm)

			elif graph_type == "multiple":
				if data_type == "GSMaP":
					if type(output) == list:
						print (output[i].shape)
						output_bound = output[i][bound[1],bound[0]]
					elif type(output) == np.ndarray:
						output_bound = output[i,bound[1],bound[0]]
				else: 	
					if type(output) == list:
						if np.isnan(np.sum(output[i])):
							continue
						else:
							output_bound = output[i][bound[0],bound[1]]
					elif type(output) == np.ndarray:
						output_bound = output[i,bound[0],bound[1]]

				fig.suptitle(self.main_title, fontsize=self.main_title_font_size,  y=self.main_title_y)
				title = sub_title[i]
				ax[i].set_title(title, fontsize=self.sub_title_font_size)
				cs = ax[i].pcolormesh(lon_map,lat_map,output_bound.reshape(shape[1],shape[0]), cmap=cmap, norm=norm)

			if i == 0:
				if (data_type == 'WRF_3km' or plot_type == 'land_only') and (color_bar == 'bias' or color_bar == 'rmse'):
					cbaxes = fig.add_axes([0.8,0.1, 0.03, 0.8])
				else:
					if color_bar == 'bias' or color_bar == 'rmse':
						cbaxes = fig.add_axes([0.87,0.1, 0.03, 0.8])
					elif  'default':
						cbaxes = fig.add_axes([0.189, 0.06, 0.647, 0.02])

			if color_bar == "default":
				cbar = fig.colorbar(cs, pad="5%", orientation="horizontal", ticks=boundaries, cax=cbaxes)
				cbar.ax.set_xticklabels(["[0,10)","[10,30)","[30,50)","[50,100)","[100,200)","[200,300)","[300,400)","[400,500)","[500,inf)"], fontsize=7)
				cbar.set_label('mm', size=12)
			elif color_bar in ["3_hourly_rainfall", "6_hourly_rainfall"]:
				cbar = fig.colorbar(cs, pad="5%", orientation="horizontal", ticks=boundaries, cax=cbaxes)
				if color_bar == "3_hourly_rainfall":
					cbar.ax.set_xticklabels(["[0,0.1)", "[0.1,1)", "[1,7.5)","[7.5,22.5)","[22.5,45)","[45,90)","[90,inf)"], fontsize=7)
				elif color_bar == "6_hourly_rainfall":
					cbar.ax.set_xticklabels(["[0,0.1)", "[0.1,1)", "[1,15)","[15,45)","[45,90)","[90,180)","[180,inf)"], fontsize=7)
				cbar.set_label('mm', size=12)
			elif color_bar in ['bias', 'rmse']:
				cbar = fig.colorbar(cs,pad="5%", ticks=boundaries, cax=cbaxes)

		plt.savefig(file_path, dpi=self.dpi)
		plt.close()

	# creates plot (single of multiple graph_type) with a provided output and save it in the file_path. The plot style is available in bar or line graph.
	# The multiple type is a multiple plot in a single graph not a multiple subplots in a single figure. A maximum of three outputs can be accepted on bar graph.
	def create_plot(self, file_path, output, cat_type=None, graph_type="single", plot_style="bar_graph"):
		directory, filename = file_path.rsplit('/', 1)

		if not os.path.exists(directory):
			os.makedirs(directory)

		fig, ax = plt.subplots(figsize=(8,8))

		x_name = self.x_name
		y_pos = np.arange(len(x_name))
		ax.set_xticks(y_pos)
		ax.set_xticklabels(x_name, fontsize=10)
		ax.set_xlabel(self.x_label)
		ax.set_xlim(-0.5, x_name.size - 0.5)

		if plot_style == "bar_graph":
			if graph_type == "single":
				bar1 = ax.bar(y_pos, output, align='center', alpha=self.alpha)
			elif graph_type == "multiple":
				w = 0.3
				color_arr = ['b', 'g', 'r']

				if output.shape[0] <= 3:
					for i in range(output.shape[0]):
						if i == 0:
							bar1 = ax.bar(y_pos-w, output[i], width=w, color=color_arr[i], align='center', alpha=self.alpha, label=self.x_legend_name[i])
						elif i == 1:
							bar2 = ax.bar(y_pos, output[i], width=w, color=color_arr[i], align='center', alpha=self.alpha, label=self.x_legend_name[i])
						elif i == 2:
							bar3 = ax.bar(y_pos+w, output[i], width=w, color=color_arr[i], align='center', alpha=self.alpha, label=self.x_legend_name[i])

				else:
					print ("Output array exceeds to 3...................")
					exit()

				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=output.shape[0])

			if len(output.shape) == 1:
				bar_arr = [bar1]
				xy_space = 0.5
			else:
				if output.shape[0] == 2:
					bar_arr = [bar1, bar2]
					xy_space = 0.15
				elif output.shape[0] == 3:
					bar_arr = [bar1, bar2, bar3]
					xy_space = 0.15

			if cat_type != None:
				for bar in bar_arr:
					for rect in bar:
						height = rect.get_height()

						if height > 2:
							ax.annotate(round(height,2), xy=(rect.xy[0]+xy_space,rect.xy[1]+0.6), xytext=(rect.xy[0]+xy_space,rect.xy[1]+0.2),rotation=90, \
								ha='center', va='center',arrowprops=dict(arrowstyle="->"))

						if np.isnan(height):
							plt.text(rect.get_x() + rect.get_width()/2.0, 0.01, 'nan', rotation=90, ha='center', va='bottom')

		elif plot_style == "line_graph":
			if graph_type == "single":
				ax.plot(y_pos, output)
			elif graph_type == "multiple":
				color_arr = ['blue', 'green', 'red', 'orange']
				marker_arr = ['o', 's', 'p', 'D']

				for i in range(output.shape[0]):
					ax.plot(y_pos, output[i], color=color_arr[i], marker ="o",label=self.x_legend_name[i])

				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=output.shape[0])

		if cat_type != None:
			cat = categorical.Categorical()

			if cat_type == 'BS':
				ax.axhline(y=1, linewidth=1, color='k', linestyle='--')
			elif cat_type == 'ETS':
				ax.axhline(y=0, linewidth=1, color='k', linestyle='-')

			ax.set_ylabel(cat.category_title(cat_type))

			if np.max(output) > 2 and plot_style == 'line_graph':
				ax.set_ylim(cat.category_min(cat_type),max(output)+0.2)
			else:
				ax.set_ylim(cat.category_min(cat_type),cat.category_max(cat_type))

		else:
			ax.set_ylabel(self.y_label)

		title = self.main_title 
		plt.title(title, fontsize=12)

		plt.savefig(file_path, dpi=self.dpi)
		plt.close()