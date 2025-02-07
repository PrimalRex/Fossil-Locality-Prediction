import gplately
from plate_model_manager import PlateModelManager

from MAIN import PFL_PATHS as pfl
from MAIN import PFL_HELPER as pflh

# Boilerplate for GPlately
pm_manager = PlateModelManager()
# Which model to init with
currentModel = "matthews2016"
# Ensure resources folder exists before writing to it
pflh.createDirectoryIfNotExist(pfl.RESOURCES_DIR)

# Utilise whichever model name from the plately API or manually edit and add model data
sourceModel = pm_manager.get_model(currentModel, data_dir=(pfl.RESOURCES_DIR / "GPLATES"))
rotationModel = sourceModel.get_rotation_model()
#topology_features = sourceModel.get_topologies()
staticPolygons = sourceModel.get_static_polygons()
model = gplately.PlateReconstruction(rotationModel, None, staticPolygons)

# Takes an array of coordinates and a timestamp and optionally a model name, returns the palaeo coordinates
def getPalaeoCoordinates(time, lon, lat, newModel="matthews2016"):
    global model, currentModel
    if newModel and newModel != currentModel:
        sourceModel = pm_manager.get_model(newModel, data_dir=(pfl.RESOURCES_DIR / "GPLATES"))
        rotationModel = sourceModel.get_rotation_model()
        staticPolygons = sourceModel.get_static_polygons()
        model = gplately.PlateReconstruction(rotationModel, None, staticPolygons)
        currentModel = newModel
    
    gpts = gplately.Points(model, lon, lat)
    rlons, rlats = gpts.reconstruct(time, return_array=True)
    return rlons, rlats

#print(getPalaeoCoordinates(245, [180], [90]))
