from math import pow, cosh, exp
import bpy
import bmesh

from sys import float_info
from mathutils import Vector

def sech(value):
	return 1.0 / cosh(value)

def sech_pow(value, factor):
	return pow(sech(value), factor)

class LayerData:
	def __init__(self):
		# Shortest distance from the node at index.
		self.distance = float_info.max
		# Node index this layer data is associated with.
		self.index = -1

# Node to represent a single vertex in the mesh.
class VNode:
	max_layers = 4

	def __init__(self, index):
		# Group data for graph traversal.
		self.group = 0
		self.index = index

		# Layers
		self.layers = [LayerData() for _ in range(self.max_layers)]
		self.temp = LayerData()

	# Clear the temp value
	def clear_temp(self):
		self.temp = LayerData()

	# Update the temp value
	def set_temp(self, dist, index):
		self.temp.distance = dist
		self.temp.index = index

	# Merge the temporary layer
	def merge_temp(self):
		# Assume that the layers are already in ascending order, maintain that order after insertion.
		loc = self.max_layers
		for i in range(len(self.layers)):
			layer = self.layers[i]

			if self.temp.distance < layer.distance:
				loc = i
				break
		
		# Early exit, no insertion
		if loc == self.max_layers:
			return

		# Shift everything down to make room for insertion
		prior = self.layers[loc]
		for i in range(loc+1, len(self.layers)):
			tmp = self.layers[i]
			self.layers[i] = prior
			prior = tmp
		
		self.layers[loc] = self.temp

		self.clear_temp()

# Graph of all the vertices
class VGraph:
	def __init__(self, bmesh_obj):

		self.mesh = bmesh_obj
		self.mesh.verts.ensure_lookup_table()
		self.mesh.edges.ensure_lookup_table()
		
		# Is index_update necessary?
		#self.mesh.verts.index_update()
		#self.mesh.edges.index_update()

		# Create the nodes
		self.nodes = [VNode(i) for i in range(len(self.mesh.verts))]
		
		# Reserved value for group ids. Monotonic increasing.
		self.tgroup = 0
	
	# Generate a new group tag for the graph traversal.
	def next_group(self):
		self.tgroup += 1
		return self.tgroup

	def branch_from(self, vertex_index, bone_index):
		# Branch out from the vertex at index, calculating the distance from that vertex to all reachable vertices.
		# Then, once the distances are calculated, merge into the layers already done.

		# Clear temp values
		for node in self.nodes:
			node.clear_temp()
		
		# Two queues:
		#   One for the current set of nodes to process.
		#   The other for the set of adjacent nodes to process in the next iteration.
		
		# Iteration ends when there are no more adjacent nodes to process (squeue is empty).
		
		queue = []
		squeue = []
		squeue.append(self.nodes[vertex_index])
		squeue[0].set_temp(0.0, bone_index)

		while len(squeue) > 0:
			# Swap the secondary with the primary
			queue, squeue = squeue, queue
			
			cgroup = self.next_group()

			# Empty the queue.
			while len(queue) > 0:
				node = queue.pop()
				ndist = node.temp.distance
				
				# Get the vertex paired with this node
				pvert = self.mesh.verts[node.index]
				
				# For each edge:
				for link in pvert.link_edges:
					overt = link.other_vert(pvert)
					onode = self.nodes[overt.index]
					
					# Add length of edge to current distance value
					summed_distance = node.temp.distance + link.calc_length()
					
					# If summed distance is less than value currently stored in adjacent node, add node to squeue for processing.
					if summed_distance < onode.temp.distance:
						onode.set_temp(summed_distance, bone_index)
						
						# Group check so we don't add duplicates to squeue.
						if onode.group != cgroup:
							squeue.append(onode)
							onode.group = cgroup

		# Update node's layer values
		for node in self.nodes:
			node.merge_temp()

# The actual operator implementation. Uses the above classes.
class OrganicWeightsOperator(bpy.types.Operator):
	bl_idname = "object.organic_weights"
	bl_label = "Calculate Organic Weights"
	bl_options = {'REGISTER', 'UNDO'}
	
	falloff: bpy.props.FloatProperty(
		name="Falloff",
		default=0.0,
		min=-10.0,
		max=10.0,
		soft_min=-10.0,
		soft_max=10.0,
		precision=2,
		step=0.1,
		subtype="FACTOR",
		description="Factor determining the rate at which the weight values falloff from their source vertices"
	)

	@classmethod
	def poll(self, context):
		return (
			context.active_object is not None and 
			context.active_object.type == 'MESH'
		)
	
	def execute(self, context):
		print("Executed the OrganicWeightsOperator!")
	
		mesh_obj = context.active_object
		mesh = mesh_obj.data
		
		prior_mode = mesh_obj.mode
	
		# Create a bmesh object for the active mesh.
		# Currently, we don't need to use any bmesh operators.
		if mesh_obj.mode == 'EDIT':
			bmesh_obj = bmesh.from_edit_mesh(mesh)
		elif mesh_obj.mode == 'OBJECT':
			bmesh_obj = bmesh.new()
			bmesh_obj.from_mesh(mesh)
		else:
			bpy.ops.object.mode_set(mode='EDIT')
			bmesh_obj = bmesh.from_edit_mesh(mesh)
		
		# For now assume that all vertex groups are being used.
		# Later, implement some kind of exclusion method, like a whitelist or a regex test.

		# Create the graph
		graph = VGraph(bmesh_obj)
		
		# So we don't have to repeatedly iterate the entire mesh to find vertex groups, precalculate the indices here:
		vgroups = []
		for i in range(len(mesh_obj.vertex_groups)):
			vgroups.append([])
		
		deform_layer = bmesh_obj.verts.layers.deform.verify()
		for vert in bmesh_obj.verts:
			dvert = vert[deform_layer]
			
			# Add vertex index to each vertex group its a part of.
			for key in dvert.keys():
				vgroups[key].append(vert.index)
		
		# Process all the vertex groups!
		for group_index, verts in enumerate(vgroups):
			for vertex_index in verts:
				graph.branch_from(vertex_index, group_index)
		
		# Calculate the power for the falloff function
		tmp_falloff_factor = exp(self.falloff)
		def falloff_func(dist):
			return sech_pow(dist, tmp_falloff_factor)
		
		# Process each node in the graph, normalizing the influence of the vertex groups according to the falloff function.
		for node in graph.nodes:
		
			# Find the valid vertex groups (possible to have different number of layers)
			valid = []
			for layer in node.layers:
				if layer.index != -1:
					valid.append(layer)
			
			# Calculate falloff
			factors = [falloff_func(layer.distance) for layer in valid]
			
			# Sum normalization factor
			norm = 0.0
			for factor in factors:
				norm += factor
			
			# Safe reciprocal
			if abs(norm) < 1e-5:
				norm = 1.0
			else:
				norm = 1.0 / norm
			
			# Assign the new weights
			vert = bmesh_obj.verts[node.index]
			dvert = vert[deform_layer]
			dvert.clear()
			for layer, factor in zip(valid, factors):
				dvert[layer.index] = factor * norm
			
			print(valid)
		
		# Cleanup and apply all the changes.
		if bmesh_obj.is_wrapped:
			bmesh.update_edit_mesh(mesh)
		else:
			bmesh_obj.to_mesh(mesh)
			mesh.update()
			bmesh_obj.free()
		
		if prior_mode != mesh_obj.mode:
			# Is this the only way to make UI update immediately when in weight paint mode?
			bpy.ops.object.editmode_toggle()
			bpy.ops.object.mode_set(mode=prior_mode)
		
		return {'FINISHED'}

def menu_func(self, context):
	self.layout.operator(OrganicWeightsOperator.bl_idname)

def register():
	bpy.utils.register_class(OrganicWeightsOperator)
	bpy.types.VIEW3D_MT_edit_mesh.append(menu_func)
	bpy.types.VIEW3D_MT_object.append(menu_func)
	bpy.types.VIEW3D_MT_paint_weight.append(menu_func)

def unregister():
	bpy.types.VIEW3D_MT_edit_mesh.remove(menu_func)
	bpy.types.VIEW3D_MT_object.remove(menu_func)
	bpy.types.VIEW3D_MT_paint_weight.remove(menu_func)
	bpy.utils.unregister_class(OrganicWeightsOperator)
