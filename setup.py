import matplotlib.pyplot as plt
import os
###################################
# Where to save the figures
PROJECT_ROOT_DIR = "/Users/seb/PycharmProjects/ACS341/Lab1"
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "../images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
#function for figure saving
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
####################################################

