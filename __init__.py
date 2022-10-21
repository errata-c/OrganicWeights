bl_info = {
    "name": "Organic Weights",
    "description": "Tools for calculating organic vertex weights", # *Tool, just one really.
    "author": "errata-c",
    "version": (0, 1, 0),
    "blender": (3, 3, 0), # Not a strict requirement?
    #"location": "Sidebar > Object > OrganicWeights",
    "warning": "",
    "category": "Object",
    "support": "TESTING", # Change this?
}

from . import calculate_weights

def register():
    calculate_weights.register()
	
def unregister():
	calculate_weights.unregister()

if __name__ == "__main__":
    register()