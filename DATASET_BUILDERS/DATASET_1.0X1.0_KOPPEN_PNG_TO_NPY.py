import numpy as np
from PIL import Image
from tqdm import tqdm

from MAIN import PFL_HELPER as pflh, PFL_PATHS as pfl

# Our Class Classification Values based on the RGB values of the Koppen Climate Map PNGs
OneHot_KoppenClasses = {
    (179, 211, 249): 0,  # Ice
    (171, 104, 17): 1,  # Cool
    (2, 251, 25): 2,  # Warm
    (255, 246, 3): 3,  # Arid
    (54, 152, 15): 4,  # Hot & Wet
}
# MAIN ------------------------------------------------------------------------

# This script generates a one-hot-encoded Koppen Climate Dataset based on the PNG files in the 1x1 Koppen Climate Maps

# Our main resource are the PNG 1x1 Koppen Climate Maps
ONE_DEG_KOPPEN_PNG_RESOURCE_DIR = pfl.RESOURCES_DIR / "1.0x1.0KoppenMaps_PNG"
# Create a directory for our stored compiled datasets
ONE_DEG_KOPPEN_RESOURCE_DIR = pfl.DATASET_DIR / "1.0x1.0KoppenMaps"
pflh.createDirectoryIfNotExist(ONE_DEG_KOPPEN_RESOURCE_DIR)

for i, file in enumerate(tqdm(pflh.getDirectoryFileNames(ONE_DEG_KOPPEN_PNG_RESOURCE_DIR), desc="Converting 1x1 Koppen PNGs to NumPy")):
        # Grabs the timestamp from the file name
        name = file.split("_")[1].split(".")[0]
        image = Image.open(ONE_DEG_KOPPEN_PNG_RESOURCE_DIR / file)

        # Convert the image to an array
        imageArray = np.array(image)
        # Convert the image to mode "P" (palette-based)
        imagePallete = image.convert("P")

        # Initialise an array for the one-hot encoded Koppen class map
        height, width = imageArray.shape[0], imageArray.shape[1]
        KoppenOneHotMap = np.zeros((height, width, len(OneHot_KoppenClasses)), dtype=int)

        # Iterate over the pixels and map the RGB colors to their corresponding class
        for y in range(height):
            for x in range(width):
                color = tuple(imagePallete.getpalette()[imageArray[y, x] * 3: imageArray[y, x] * 3 + 3])
                if color in OneHot_KoppenClasses:
                    class_idx = OneHot_KoppenClasses[color]
                    KoppenOneHotMap[y, x, class_idx] = 1

        # print("Shape of one-hot encoded Koppen map:", KoppenOneHotMap.shape)
        # plt.figure(figsize=(20, 10))
        # plt.imshow(KoppenOneHotMap[:, :, 1], cmap='viridis', interpolation='nearest')
        # plt.title("Köppen Climate Map - Ice Class")
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

        np.save(ONE_DEG_KOPPEN_RESOURCE_DIR / f"{name}_Koppen_361.npy", KoppenOneHotMap)
