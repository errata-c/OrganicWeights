from math import pow, cosh, exp
import bpy
import bmesh

from sys import float_info
from mathutils import Vector

def sech(value):
	return 1.0 / cosh(value)

def sech_pow(value, factor):
	return pow(sech(value), factor)

# Node to represent a single vertex in the mesh.
class VNode:
	max_layers = 4

	def __init__(self, index):
		# Tag for graph traversal
		self.tag = 0
		self.index = index

		# Layer data
		self.distances = []
		self.groups = []

	def in_range(self, dist):
		if len(self.distances) < 4:
			return True
		
		return self.distances[-1] > dist
	
	def find_group(self, group):
		for i in range(len(self.groups)):
			if self.groups[i] == group:
				return i
		return len(self.groups)
	
	def get_distance(self, group):
		return self.distances[self.groups.index(group)]
	
	def sort(self):
		p = len(self.distances)-2
		while p >= 0 and self.distances[p+1] < self.distances[p]:
			tmp_dist = self.distances[p]
			tmp_group = self.groups[p]
			
			self.distances[p] = self.distances[p+1]
			self.groups[p] = self.groups[p+1]
			
			self.distances[p+1] = tmp_dist
			self.groups[p+1] = tmp_group
	
	# Make sure max layers is not exceeded
	def trim_excess(self):
		if len(self.distances) > self.max_layers:
			self.distances.pop()
			self.groups.pop()
	
	# Prepare for a new graph traversal
	def initialize_group(self, group):
		self.distances.insert(0, 0.0)
		self.groups.insert(0, group)
		
		self.trim_excess() 
	
	# Update or insert a value
	def insert(self, dist, group):
		loc = self.find_group(group)
		if loc == len(self.groups):
			# If the group is not already in the list, append the new values
			
			self.distances.append(dist)
			self.groups.append(group)
		else:
			# Otherwise, just update the old value
			
			self.distances[loc] = min(self.distances[loc], dist)
		
		# Maintain ordering
		self.sort()
		
		self.trim_excess()
		

# Graph of all the vertices
class VGraph:
	def __init__(self, bmesh_obj):

		self.mesh = bmesh_obj
		self.mesh.verts.ensure_lookup_table()
		self.mesh.edges.ensure_lookup_table()
		
		# Is index_update necessary? I'm just gonna do it, we can check later if necessary.
		self.mesh.verts.index_update()
		self.mesh.edges.index_update()

		# Create the nodes
		self.nodes = [VNode(i) for i in range(len(self.mesh.verts))]
		
		# Reserve tag value for graph traversal
		self.ntag = 0
	
	# Generate a new group tag for the graph traversal.
	def next_tag(self):
		self.ntag += 1
		return self.ntag

	def branch_from(self, vertex_index, group):
		# Branch out from the vertex at index, calculating the distance from that vertex to all reachable vertices.
		# Then, once the distances are calculated, merge into the layers already done.
		
		# Two queues:
		#   One for the current set of nodes to process.
		#   The other for the set of adjacent nodes to process in the next iteration.
		
		# Iteration ends when there are no more adjacent nodes to process (squeue is empty).
		
		queue = []
		squeue = []
		
		first_node = self.nodes[vertex_index]
		first_node.initialize_group(group)
		
		squeue.append(first_node)
		
		while len(squeue) > 0:
			# Swap the secondary with the primary
			queue, squeue = squeue, queue

			# Get a tag value
			ctag = self.next_tag()

			# Empty the queue
			while len(queue) > 0:
				node = queue.pop()
				dist = node.get_distance(group)
				
				# Get the vertex paired with this node
				pvert = self.mesh.verts[node.index]
				
				# For each edge:
				for link in pvert.link_edges:
					overt = link.other_vert(pvert)
					onode = self.nodes[overt.index]
					
					# Add length of edge to current distance value
					summed_distance = dist + link.calc_length()
					
					# If the value will fit, add it in.
					if onode.in_range(summed_distance):
						onode.insert(summed_distance, group)
						
						# Tag check so we don't add duplicates to squeue.
						if onode.tag != ctag:
							squeue.append(onode)
							onode.tag = ctag

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
			print(f"Processing group {group_index} out of {len(vgroups)}.")
			
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
