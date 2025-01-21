import gplately
from plate_model_manager import PlateModelManager

from MAIN import PFL_PATHS as pfl

# Boilerplate for GPlately, references Scotese2016 model to match existing dataset source
pm_manager = PlateModelManager()
sourceModel = pm_manager.get_model("Matthews2016", data_dir=(pfl.RESOURCES_DIR / "GPLATES"))
rotationModel = sourceModel.get_rotation_model()
# topology_features = muller2019_model.get_topologies()
staticPolygons = sourceModel.get_static_polygons()
model = gplately.PlateReconstruction(rotationModel, None, staticPolygons)

# Takes an array of coordinates and a timestamp and uses the predefined model to return the palaeo coordinates
def getPalaeoCoordinates(time, lon, lat):
    gpts = gplately.Points(model, lon, lat)
    rlons, rlats = gpts.reconstruct(time, return_array=True)
    return rlons, rlats

#print(getPalaeoCoordinates(245, [180], [90]))
